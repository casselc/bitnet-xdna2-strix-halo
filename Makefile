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

.PHONY: all check check-cpu check-npu clean
all: $(BUILD)/test_i2s_packing $(BUILD)/test_coordinates \
     $(BUILD)/test_i2s_realdata $(BUILD)/test_xdna_gemm \
     $(BUILD)/npu_probe $(BUILD)/npu_gemm_bench $(BUILD)/npu_switch_cost

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
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB) -lm
$(BUILD)/npu_probe: tools/npu_probe.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB)
$(BUILD)/npu_gemm_bench: tools/npu_gemm_bench.cpp $(RT_CXX) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB)
$(BUILD)/npu_switch_cost: tools/npu_switch_cost.cpp $(RT_CXX) | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(XRTINC) $(XRTLIB)

check-cpu: $(BUILD)/test_i2s_packing $(BUILD)/test_coordinates $(BUILD)/test_i2s_realdata
	@echo "=== CPU-only tests (no NPU needed) ==="
	@$(BUILD)/test_i2s_packing
	@echo
	@$(BUILD)/test_coordinates
	@echo
	@$(BUILD)/test_i2s_realdata

check-npu: $(BUILD)/test_xdna_gemm
	@echo "=== NPU tests ==="
	@$(BUILD)/test_xdna_gemm

check: check-cpu check-npu

clean:
	rm -rf $(BUILD)
