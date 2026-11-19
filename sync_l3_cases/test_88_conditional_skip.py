"""L3 88: 条件跳轮 / 动态 trip.

代表用例：testing.ascend.language.test_tilelang_ascend_autosync_device::test_conditional_skip_then_normal_launch_has_no_residual_token
"""

import pytest

import torch

import tilelang

import tilelang.ascend.language as T

import tilelang.testing

BLOCKS = 8

TILE = 1024

def _guarded_loop(N):
    iterations = N // (TILE * BLOCKS)
    assert N == TILE * BLOCKS * iterations

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
        count: T.int32,
    ):
        with T.Kernel(BLOCKS) as bx:
            a = T.alloc_shared((TILE,), "float32")
            b = T.alloc_shared((TILE,), "float32")
            c = T.alloc_shared((TILE,), "float32")
            T.annotate_buffer_versions({a: 2, b: 2, c: 2})
            for step in T.Pipelined(iterations, num_stages=2):
                if step < count:
                    begin = (step * BLOCKS + bx) * TILE
                    T.copy(A[begin : begin + TILE], a)
                    T.copy(B[begin : begin + TILE], b)
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(TILE):
                            c[i] = a[i] + b[i]
                    T.copy(c, C[begin : begin + TILE])

    return main

def test_conditional_skip_then_normal_launch_has_no_residual_token():
    N = TILE * BLOCKS * 4
    kernel = tilelang.compile(_guarded_loop(N), target="ascend")
    a = torch.randn(N, device="npu")
    b = torch.randn(N, device="npu")
    out = torch.full((N,), -7.0, device="npu")
    kernel(a, b, out, 0)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.full_like(out, -7.0), atol=0, rtol=0)
    kernel(a, b, out, 4)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, atol=1e-5, rtol=1e-5)
