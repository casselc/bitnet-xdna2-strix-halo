//===- mm_i8_i4.cc — int8 activations x int4 ternary weights -> int32 -------===//
//
// BitNet's weights are {-1,0,+1}, which fit int4 exactly. Storing them as int4
// rather than int8 halves the weight bytes crossing DMA (1843 -> 922 MiB
// resident for BitNet-2B), and on aie2p the 4-bit expansion is done by the
// load-store unit -- disassembly of a compiled kernel shows
//
//     vldb.unpack  y0, unpacksign1, [p1, #0x0]
//
// so it costs no MAC or vector-ALU slots. mlir-aie's stock mm.cc cannot express
// this: it hardcodes aie::mmul<r,s,t,T_in,T_in,accauto>, the SAME type for both
// operands, and IRON's kernels.mm() exposes no int4 combination.
//
// On aie2p (__AIE_ARCH__ == 21) aie::mmul<4,16,16,int8,int4> lowers to a single
// native mac_4x16_16x16_conf: 1024 MACs/instruction against mac_8x8_8x8's 512.
// Whether that issues at 1x or 2x the rate is exactly what this kernel exists to
// measure -- see artifacts/kernels/int4_investigation.md.
//
// Layout follows mm.cc: A, B and C are all tile-major, with the (r,s,t) micro-
// tiles contiguous. B carries s*t = 256 int4 values = 128 bytes per micro-tile.
//===----------------------------------------------------------------------===//

#define NOCPP
#include <stdint.h>
#include <type_traits>
#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

// r x s x t for the aie2p int8 x int4 mmul.
constexpr unsigned MM_R = 4;
constexpr unsigned MM_S = 16;
constexpr unsigned MM_T = 16;

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void matmul_i8i4_core(const int8 *__restrict pA,
                                    const int8 *__restrict pB,
                                    int32 *__restrict pC) {
  using MMUL = aie::mmul<MM_R, MM_S, MM_T, int8, int4, accauto>;
  // B strides are computed in BYTES, not in int4 elements. Pointer arithmetic on
  // a 4-bit type does not advance by half-bytes -- stepping an int4* by size_B
  // moves size_B *bytes*, i.e. twice as far as intended. That is invisible
  // within a single mmul tile (one load, no stride) and only corrupts the K
  // blocks after the first, which is exactly how it presented: B[p,0] was exact
  // for p < 16 and wrong from p = 16 on.
  constexpr unsigned B_TILE_BYTES = MMUL::size_B / 2;
  event0();

  for (unsigned z = 0; z < rowA; z++)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      int32 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j++) {
        const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const int8 *__restrict pB1 = pB + j * B_TILE_BYTES;

        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<int4, MMUL::size_B> B0 =
            aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(pB1));
        pB1 += B_TILE_BYTES * colB;

        MMUL C00;
        C00.mul(A0, B0);

        for (unsigned i = 1; i < colA; i++)
          chess_prepare_for_pipelining chess_loop_range(3, ) {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            B0 = aie::load_v<MMUL::size_B>(reinterpret_cast<const int4 *>(pB1));
            pB1 += B_TILE_BYTES * colB;
            C00.mac(A0, B0);
          }

        aie::store_v(pC1, C00.template to_vector<int32>());
        pC1 += MMUL::size_C;
      }
    }

  event1();
}

template <typename T_out, unsigned M, unsigned N>
static inline void zero_core(T_out *__restrict pC) {
  const aie::vector<T_out, 32> z = aie::zeros<T_out, 32>();
  T_out *__restrict p = pC;
  for (unsigned i = 0; i < (M * N) / 32; i++)
    chess_prepare_for_pipelining {
      aie::store_v(p, z);
      p += 32;
    }
}

// Matched int8 x int8 path. Same loop nest, same accumulator handling, same
// REPEAT; only the mmul operand type and (r,s,t) differ. Anything else that
// changed between the two would contaminate the throughput ratio.
constexpr unsigned M8_R = 8, M8_S = 8, M8_T = 8;

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void matmul_i8i8_core(const int8 *__restrict pA,
                                    const int8 *__restrict pB,
                                    int32 *__restrict pC) {
  using MMUL = aie::mmul<M8_R, M8_S, M8_T, int8, int8, accauto>;
  event0();
  for (unsigned z = 0; z < rowA; z++)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      int32 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      for (unsigned j = 0; j < colB; j++) {
        const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const int8 *__restrict pB1 = pB + j * MMUL::size_B;
        aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B * colB;
        MMUL C00;
        C00.mul(A0, B0);
        for (unsigned i = 1; i < colA; i++)
          chess_prepare_for_pipelining chess_loop_range(3, ) {
            A0 = aie::load_v<MMUL::size_A>(pA1);
            pA1 += MMUL::size_A;
            B0 = aie::load_v<MMUL::size_B>(pB1);
            pB1 += MMUL::size_B * colB;
            C00.mac(A0, B0);
          }
        aie::store_v(pC1, C00.template to_vector<int32>());
        pC1 += MMUL::size_C;
      }
    }
  event1();
}

#ifndef REPEAT
#define REPEAT 1
#endif

extern "C" {

void matmul_i8_i8_i32(int8 *a_in, int8 *b_in, int32 *c_out) {
  for (int r = 0; r < REPEAT; r++)
    matmul_i8i8_core<DIM_M / M8_R, DIM_K / M8_S, DIM_N / M8_T>(a_in, b_in, c_out);
}

// B arrives as a byte buffer: IRON/numpy have no int4 dtype, so the host packs
// two 4-bit weights per byte and the kernel reinterprets. The DMA moves half the
// bytes of an int8 weight tile; the load-store unit does the widening.
void matmul_i8_i4_i32(int8 *a_in, int8 *b_in, int32 *c_out) {
  for (int r = 0; r < REPEAT; r++)
    matmul_i8i4_core<DIM_M / MM_R, DIM_K / MM_S, DIM_N / MM_T>(a_in, b_in, c_out);
}

void zero_i32(int32 *c_out) { zero_core<int32, DIM_M, DIM_N>(c_out); }

}
