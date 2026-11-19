"""L3 87: 流水首尾与短循环.

代表用例：testing.ascend.language.test_tilelang_ascend_autosync_device::test_short_pipeline_does_not_hang_or_overrun
"""

import pytest

import torch

import tilelang

import tilelang.ascend.language as T

import tilelang.testing

BLOCKS = 8

TILE = 1024

def _short_loop(N, trips):
    iterations = N // (TILE * BLOCKS)
    assert N == TILE * BLOCKS * iterations

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
        scratch: T.Tensor((BLOCKS,), "float32"),
    ):
        with T.Kernel(BLOCKS) as bx:
            a = T.alloc_shared((TILE,), "float32")
            b = T.alloc_shared((TILE,), "float32")
            c = T.alloc_shared((TILE,), "float32")
            T.annotate_buffer_versions({a: 2, b: 2, c: 2})
            # Keep the kernel non-empty so a zero-trip loop still has launch args.
            scratch[bx] = A[bx]
            for step in T.Pipelined(trips, num_stages=2):
                begin = (step * BLOCKS + bx) * TILE
                T.copy(A[begin : begin + TILE], a)
                T.copy(B[begin : begin + TILE], b)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(TILE):
                        c[i] = a[i] + b[i]
                T.copy(c, C[begin : begin + TILE])

    return main

@pytest.mark.parametrize("trips", [0, 1, 2, 3], ids=["zero", "one", "two", "three"])
def test_short_pipeline_does_not_hang_or_overrun(trips):
    N = TILE * BLOCKS * 4
    kernel = tilelang.compile(_short_loop(N, trips), target="ascend")
    a = torch.randn(N, device="npu")
    b = torch.randn(N, device="npu")
    result = torch.full((N,), -7.0, device="npu")
    kernel(a, b, result, torch.empty(BLOCKS, device="npu"))
    torch.npu.synchronize()
    if trips == 0:
        torch.testing.assert_close(result, torch.full_like(result, -7.0), atol=0, rtol=0)
    else:
        expected = (a + b).clone()
        expected[trips * BLOCKS * TILE :] = -7.0
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
