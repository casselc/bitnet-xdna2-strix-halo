# Strix Halo / XDNA2 BitNet hybrid MVP
#
# Everything here assumes the environment described in artifacts/environment.md.
# `make check` needs the NPU; `make check-cpu` does not.

CC      := clang
CXX     := clang++
CFLAGS  := -O2 -Wall -Wextra
CXXFLAGS:= -O2 -std=c++17 -Wall
XRTINC  := -I/usr/include/xrt -I$(CURDIR)/.localdeps/usr/include
XRTLIB  := -lxrt_coreutil
BUILD   := build

RT_C    := runtime/bitnet_i2s.c runtime/bitnet_coord.c
RT_CXX  := runtime/xdna_gemm.cpp

.PHONY: all check check-cpu check-npu check-patch clean
all: $(BUILD)/test_i2s_packing $(BUILD)/test_coordinates \
     $(BUILD)/test_i2s_realdata $(BUILD)/test_xdna_gemm \
     $(BUILD)/npu_probe $(BUILD)/npu_gemm_bench $(BUILD)/npu_switch_cost \
     $(BUILD)/test_xdna_shapes $(BUILD)/test_xdna_concurrent $(BUILD)/npu_stress

$(BUILD):
	@mkdir -p $(BUILD)

# --- CPU-only: no NPU required -------------------------------------------
$(BUILD)/test_i2s_packing: tests/test_i2s_packing.c runtime/bitnet_i2s.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm
$(BUILD)/test_i2s_realdata: tests/test_i2s_realdata.c runtime/bitnet_i2s.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm
$(BUILD)/test_coordinates: tests/test_coordinates.c runtime/bitnet_coord.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^

# --- NPU required ---------------------------------------------------------
$(BUILD)/test_xdna_gemm: tests/test_xdna_gemm.cpp $(RT_CXX) runtime/bitnet_i2s.c | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB) -lm -lpthread
$(BUILD)/test_xdna_shapes: tests/test_xdna_shapes.cpp runtime/bitnet_xdna.cpp $(RT_CXX) runtime/bitnet_i2s.c | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB) -lm -lpthread

$(BUILD)/test_xdna_concurrent: tests/test_xdna_concurrent.cpp runtime/bitnet_xdna.cpp $(RT_CXX) runtime/bitnet_i2s.c | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB) -lm -lpthread
$(BUILD)/npu_two_context: tools/npu_two_context.cpp $(RT_CXX) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB) -lm -lpthread
$(BUILD)/npu_stress: tools/npu_stress.cpp runtime/bitnet_xdna.cpp $(RT_CXX) runtime/bitnet_i2s.c | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB) -lm -lpthread
$(BUILD)/npu_probe: tools/npu_probe.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB)
$(BUILD)/npu_gemm_bench: tools/npu_gemm_bench.cpp $(RT_CXX) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB)
$(BUILD)/npu_switch_cost: tools/npu_switch_cost.cpp $(RT_CXX) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB)

TENSOR := artifacts/correctness/tensors/attn_q_l0.packed
GGUF   := models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

# The tests validate against a real weight slice, which is deliberately not
# checked in (it is model data). Extract it on demand so `make check` works on a
# fresh clone instead of failing with "cannot open ...".
$(TENSOR):
	@test -f $(GGUF) || { echo "missing $(GGUF) -- see artifacts/reproduce.md step 1"; exit 1; }
	@mkdir -p $(dir $@)
	.venv/bin/python tools/gguf_extract.py $(GGUF) blk.0.attn_q.weight artifacts/correctness/tensors/attn_q_l0

check-cpu: $(BUILD)/test_i2s_packing $(BUILD)/test_coordinates $(BUILD)/test_i2s_realdata $(TENSOR)
	@echo "=== CPU-only tests (no NPU needed) ==="
	@$(BUILD)/test_i2s_packing
	@echo
	@$(BUILD)/test_coordinates
	@echo
	@$(BUILD)/test_i2s_realdata

SHAPE_TENSORS := artifacts/correctness/tensors/ffn_gate_l0.packed artifacts/correctness/tensors/ffn_down_l0.packed
$(SHAPE_TENSORS): $(GGUF)
	@mkdir -p $(dir $@)
	.venv/bin/python tools/gguf_extract.py $(GGUF) blk.0.ffn_gate.weight artifacts/correctness/tensors/ffn_gate_l0
	.venv/bin/python tools/gguf_extract.py $(GGUF) blk.0.ffn_down.weight artifacts/correctness/tensors/ffn_down_l0

check-npu: $(BUILD)/test_xdna_gemm $(BUILD)/test_xdna_shapes $(BUILD)/test_xdna_concurrent $(TENSOR) $(SHAPE_TENSORS)
	@echo "=== NPU tests ==="
	@$(BUILD)/test_xdna_gemm
	@echo
	@BITNET_XDNA=1 BITNET_XDNA_ARTIFACTS=$(CURDIR)/artifacts/xclbin-tuned $(BUILD)/test_xdna_shapes
	@echo
	@BITNET_XDNA=1 BITNET_XDNA_ARTIFACTS=$(CURDIR)/artifacts/xclbin-tuned $(BUILD)/test_xdna_concurrent

# --- patch reproducibility ------------------------------------------------
# patches/001-bitnet-xdna.patch has silently gone stale more than once, which
# makes a benchmark from build-xdnaN non-reproducible: the checked-in patch would
# have produced different source than the one measured. This regenerates the diff
# from the pinned BitNet tree and fails if it differs, and verifies the patch
# still applies to a pristine checkout of that tree.
LLAMA := refs/BitNet/3rdparty/llama.cpp
check-patch:
	@echo "=== patch reproducibility ==="
	@cd $(LLAMA) && git diff > /tmp/.patchcheck.live
	@if diff -q /tmp/.patchcheck.live patches/001-bitnet-xdna.patch >/dev/null; then \
	    echo "  ok  checked-in patch matches the working tree being built"; \
	else \
	    echo "  FAIL checked-in patch differs from the source actually built:"; \
	    diff -u patches/001-bitnet-xdna.patch /tmp/.patchcheck.live | head -40; \
	    rm -f /tmp/.patchcheck.live; exit 1; \
	fi
	@rm -rf /tmp/.patchcheck.tree && mkdir -p /tmp/.patchcheck.tree
	@cd $(LLAMA) && git archive HEAD | tar -x -C /tmp/.patchcheck.tree
	@cd /tmp/.patchcheck.tree && git apply --check $(CURDIR)/patches/001-bitnet-xdna.patch \
	    && echo "  ok  patch applies cleanly to a pristine checkout of the pinned tree" \
	    || { echo "  FAIL patch does not apply to the pinned tree"; exit 1; }
	@rm -rf /tmp/.patchcheck.tree /tmp/.patchcheck.live

check: check-patch check-cpu check-npu

clean:
	rm -rf $(BUILD)
