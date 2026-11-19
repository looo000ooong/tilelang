"""L3 84: 跨核同步.

代表用例：testing.ascend.auto_schedule.test_schedule_integration::test_subcore_gemm_results
"""

import pytest

import tilelang

import tilelang.testing

import tilelang.ascend.language as T

from tilelang import tvm

from tvm import tirx

from testing.ascend._ir import nodes, calls, allocated_buffer

def _make_subcore_gemm(pattern, steps, unit_flag=0):
    @T.prim_func
    def main(
        A: T.Tensor((steps, 32, 32), "bfloat16"),
        B: T.Tensor((32, 32), "bfloat16"),
        C: T.Tensor((steps, 32, 32), "float32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            a_l1 = T.alloc_l1((32, 32), "bfloat16")
            b_l1 = T.alloc_l1((32, 32), "bfloat16")
            accum = T.alloc_l0c((32, 32), "float32")
            dq = T.alloc_shared((32, 32), "bfloat16")
            half = T.alloc_shared((16, 32), "bfloat16")
            epi = T.alloc_shared((32, 32), "float32")
            T.annotate_buffer_versions({epi: (2, "counter")})
            T.copy(B, b_l1)
            for i in T.Pipelined(steps, num_stages=2):
                if pattern == "epilogue":
                    T.copy(A[i, :, :], a_l1)
                elif pattern == "halves":
                    T.copy(A[i, sid * 16 : (sid + 1) * 16, :], half)
                    with T.SimdVF():
                        value = T.simd.vld(half[0, 0])
                        T.simd.vsts(half[0, 0], value)
                    T.copy(half, a_l1[sid * 16 : (sid + 1) * 16, :])
                elif pattern == "fixed":
                    if sid == 0:
                        T.copy(A[i, :, :], dq)
                        with T.SimdVF():
                            value = T.simd.vld(dq[0, 0])
                            T.simd.vsts(dq[0, 0], value)
                        T.copy(dq, a_l1)
                else:
                    if sid == i % 2:
                        T.copy(A[i, :, :], dq)
                        with T.SimdVF():
                            value = T.simd.vld(dq[0, 0])
                            T.simd.vsts(dq[0, 0], value)
                        T.copy(dq, a_l1)
                T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True, unit_flag_ctrl=unit_flag)
                if pattern == "fixed":
                    if sid == 1:
                        T.copy(accum, epi, unit_flag_ctrl=unit_flag)
                        with T.SimdVF():
                            value = T.simd.vld(epi[0, 0])
                            T.simd.vsts(epi[0, 0], value)
                        T.copy(epi, C[i, :, :])
                else:
                    if sid == (i + 1) % 2:
                        T.copy(accum, epi, unit_flag_ctrl=unit_flag)
                        with T.SimdVF():
                            value = T.simd.vld(epi[0, 0])
                            T.simd.vsts(epi[0, 0], value)
                        T.copy(epi, C[i, :, :])

    return main

@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "pattern, steps, unit_flag",
    [
        ("epilogue", 1, 0),
        ("epilogue", 5, 0),
        ("fixed", 5, 0),
        ("halves", 5, 0),
        ("switch", 5, 0),
        ("epilogue", 5, 3),
        ("fixed", 5, 3),
    ],
)
def test_subcore_gemm_results(pattern, steps, unit_flag):
    import torch

    torch.manual_seed(23)
    a = torch.randn(steps, 32, 32, dtype=torch.bfloat16).npu()
    b = torch.randn(32, 32, dtype=torch.bfloat16).npu()
    output = tilelang.compile(_make_subcore_gemm(pattern, steps, unit_flag), target="ascend", out_idx=-1)(a, b)
    torch.testing.assert_close(output.cpu(), a.cpu().float() @ b.cpu().float().T, rtol=0.01, atol=0.02)
