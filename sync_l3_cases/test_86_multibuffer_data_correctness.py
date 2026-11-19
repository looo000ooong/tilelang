"""L3 86: 多缓冲流水数据正确性.

代表用例：testing.ascend.language.test_tilelang_ascend_autosync_device::test_multibuffer_versions_match_single_buffer
"""

import pytest

import torch

import tilelang

import tilelang.ascend.language as T

import tilelang.testing

BLOCKS = 8

TILE = 1024

def _vector_add(N, versions):
    iterations = N // (TILE * BLOCKS)
    assert N == TILE * BLOCKS * iterations

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(BLOCKS) as bx:
            a = T.alloc_shared((TILE,), "float32")
            b = T.alloc_shared((TILE,), "float32")
            c = T.alloc_shared((TILE,), "float32")
            T.annotate_buffer_versions({a: versions, b: versions, c: versions})
            for step in T.Pipelined(iterations, num_stages=versions):
                begin = (step * BLOCKS + bx) * TILE
                T.copy(A[begin : begin + TILE], a)
                T.copy(B[begin : begin + TILE], b)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(TILE):
                        c[i] = a[i] + b[i]
                T.copy(c, C[begin : begin + TILE])

    return main

@pytest.mark.parametrize("versions", [1, 2, 3], ids=["single", "double", "triple"])
def test_multibuffer_versions_match_single_buffer(versions):
    N = TILE * BLOCKS * 4
    kernel = tilelang.compile(_vector_add(N, versions), out_idx=-1, target="ascend")
    a = torch.randn(N, device="npu")
    b = torch.randn(N, device="npu")
    torch.testing.assert_close(kernel(a, b), a + b, atol=1e-5, rtol=1e-5)
