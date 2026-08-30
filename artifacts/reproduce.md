# Exact reproduction

Every command below was run on the machine described in `environment.md`.
Paths are relative to the repository root.

## 0. Privileged setup (once)

```bash
sudo apt update
sudo apt install -y libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu \
                    python3-xrt cmake clang ninja-build python3-pip python3-venv pkg-config

# device permission without requiring a re-login
echo 'SUBSYSTEM=="accel", KERNEL=="accel0", MODE="0666"' | sudo tee /etc/udev/rules.d/99-amdxdna.rules
sudo udevadm control --reload && sudo udevadm trigger

# memlock: XRT mmaps 64 MiB locked; the 8 MiB default fails with EAGAIN
printf '* soft memlock unlimited\n* hard memlock unlimited\n' | sudo tee /etc/security/limits.d/99-npu-memlock.conf
# ...and raise it on the already-running shell (limits.d only applies at login):
sudo prlimit --pid <your shell's pid> --memlock=unlimited:unlimited
```

Verify:
```bash
ulimit -l                # expect: unlimited
xrt-smi examine          # expect: RyzenAI-npu5 | aie2p | 6x8, FW 1.1.2.65
xrt-smi examine -r platform | grep Columns    # expect: Total Columns : 8
```

`uuid-dev` is not installed system-wide; the XRT headers need `uuid/uuid.h`, so
it is unpacked locally (no root) and the Makefile points at it:
```bash
mkdir -p .localdeps && cd .localdeps && apt-get download uuid-dev && dpkg-deb -x uuid-dev_*.deb .
```

## 1. Toolchain and model

```bash
python3 -m venv --system-site-packages .venv        # --system-site-packages keeps apt's pyxrt visible
.venv/bin/pip install huggingface_hub numpy
.venv/bin/pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.2
.venv/bin/pip install "llvm-aie==21.0.0.2026080301+c9c5ecb7" \
    -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

git -C refs/BitNet submodule update --init --recursive     # isHuangXin/llama.cpp @ 390c307
.venv/bin/hf download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
sha256sum models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# 4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
```

## 2. Patch and build BitNet

`patches/001-bitnet-xdna.patch` is REQUIRED: the fork's I2_S fast path
references `src1_cont`, which is only declared inside `#if GGML_USE_LLAMAFILE`,
so the file does not compile with `-DGGML_LLAMAFILE=OFF`. We need llamafile off
because `llamafile_sgemm_i2s` otherwise preempts the dispatch site we hook.

```bash
cd refs/BitNet
# CPU-only reference build
cmake -B build -G Ninja -DBITNET_X86_TL2=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DGGML_LLAMAFILE=OFF
cmake --build build -j32

# hybrid build (same flags + the XDNA runtime)
R=$(cd ../.. && pwd)
cmake -B build-xdna -G Ninja -DBITNET_X86_TL2=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DGGML_LLAMAFILE=OFF \
      -DBITNET_XDNA_RUNTIME_DIR=$R/runtime \
      -DBITNET_XDNA_XRT_INCLUDE="/usr/include/xrt;$R/.localdeps/usr/include"
cmake --build build-xdna -j32
```

## 3. Compile the NPU kernels (ahead of time)

mlir-aie is used only as an offline compiler; the inference path links the C++
XRT API and never touches Python.

```bash
for KN in "2560 2560 64" "2560 3456 48" "6912 2560 64"; do
  set -- $KN
  .venv/bin/python npu/ref/whole_array.py -M 512 -K $1 -N $2 -m 64 -k 64 -n $3 \
    --dtype_in i8 --dtype_out i32 --n-aie-cols 8 \
    --xclbin-path artifacts/xclbin/mm_M512_K$1_N$2.xclbin \
    --insts-path  artifacts/xclbin/mm_M512_K$1_N$2.insts.bin
done
```
(`npu/ref/whole_array.py` is fetched verbatim from `Xilinx/mlir-aie@v1.4.2`,
`programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`.)

## 4. Tests

```bash
make check          # everything; needs the NPU
make check-cpu      # packing, coordinates, real-GGUF layout -- no NPU needed
```

## 5. Benchmarks

```bash
export BITNET_XDNA_ARTIFACTS=$PWD/artifacts/xclbin
M=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# CPU only
BITNET_XDNA=0 refs/BitNet/build-xdna/bin/llama-bench -m $M -p 128,512,2048,3968 -n 32 -t 16 -ngl 0 -r 2

# NPU-assisted prefill + CPU decode
BITNET_XDNA=1 BITNET_XDNA_STATS=1 \
  refs/BitNet/build-xdna/bin/llama-bench -m $M -p 128,512,2048,3968 -n 32 -t 16 -ngl 0 -r 2

# proof that decode never touches the NPU
BITNET_XDNA=1 BITNET_XDNA_STATS=1 refs/BitNet/build-xdna/bin/llama-bench -m $M -p 512 -n 0   -t 16 -ngl 0 -r 1
BITNET_XDNA=1 BITNET_XDNA_STATS=1 refs/BitNet/build-xdna/bin/llama-bench -m $M -p 0   -n 128 -t 16 -ngl 0 -r 1

# kernel-level breakdown
build/npu_probe          # BO alloc / sync costs, no kernel involved
build/npu_gemm_bench     # dispatch cost vs number of resident weight buffers
build/npu_switch_cost    # kernel time vs xclbin-switching overhead
```

## 6. Output equivalence

```bash
{ cat artifacts/correctness/controller_prompt.txt; printf '\n/exit\n'; } | \
  BITNET_XDNA=0 refs/BitNet/build-xdna/bin/llama-cli -m $M -n 32 -t 16 -ngl 0 --temp 0 --seed 42

{ cat artifacts/correctness/controller_prompt.txt; printf '\n/exit\n'; } | \
  BITNET_XDNA=1 refs/BitNet/build-xdna/bin/llama-cli -m $M -n 32 -t 16 -ngl 0 --temp 0 --seed 42
```
Note `llama-cli` in this fork is an interactive REPL: `-no-cnv` is unsupported and
`llama-completion` crashes in `common_chat_format_example`, so the prompt is piped
and terminated with `/exit`.
