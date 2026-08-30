# CPU oracle baseline

Model: microsoft/BitNet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf
sha256: 4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162
Build:  clang 21.1.8, -DGGML_LLAMAFILE=OFF, -DBITNET_X86_TL2=OFF (i2_s kernels)
        + patches/001-bitnet-xdna.patch (required to compile at all)
Host:   AMD RYZEN AI MAX+ 395, 16C/32T

## llama-bench, 16 threads (see thread sweep below for why 16)
```
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| bitnet-b1.58 2B Q1_0           |   1.10 GiB |     2.41 B | CPU        |      16 |           pp128 |      1081.18 ± 40.32 |
| bitnet-b1.58 2B Q1_0           |   1.10 GiB |     2.41 B | CPU        |      16 |           pp512 |       1277.33 ± 0.57 |
| bitnet-b1.58 2B Q1_0           |   1.10 GiB |     2.41 B | CPU        |      16 |          pp2048 |      1029.69 ± 10.10 |
| bitnet-b1.58 2B Q1_0           |   1.10 GiB |     2.41 B | CPU        |      16 |          pp3968 |        804.82 ± 1.82 |
| bitnet-b1.58 2B Q1_0           |   1.10 GiB |     2.41 B | CPU        |      16 |            tg32 |         79.80 ± 0.82 |

build: 390c30775 (9918)
```
