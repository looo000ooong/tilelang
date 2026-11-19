"""L3 89: 手工多缓冲正确性.

代表用例：testing.ascend.language.test_tilelang_ascend_autosync_device::test_manual_three_slot_buffer_is_numerically_correct
"""

import pytest

import torch

import tilelang

import tilelang.ascend.language as T

import tilelang.testing

BLOCKS = 8

TILE = 1024

def _manual_slots(N, rotate):
    iterations = N // (TILE * BLOCKS)
    assert N == TILE * BLOCKS * iterations

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(BLOCKS) as bx:
            a = T.alloc_shared((3, TILE), "float32")
            b = T.alloc_shared((3, TILE), "float32")
            c = T.alloc_shared((3, TILE), "float32")
            T.annotate_manual_multi_buffer({a: 3, b: 3, c: 3})
            for step in T.Pipelined(iterations, num_stages=3):
                slot = step % 3 if rotate else 0
                begin = (step * BLOCKS + bx) * TILE
                T.copy(A[begin : begin + TILE], a[slot, :])
                T.copy(B[begin : begin + TILE], b[slot, :])
                with T.SimtVF(threads=256):
                    for i in T.Parallel(TILE):
                        c[slot, i] = a[slot, i] + b[slot, i]
                T.copy(c[slot, :], C[begin : begin + TILE])

    return main

@pytest.mark.parametrize("rotate", [True, False], ids=["rotating", "fixed-slot"])
def test_manual_three_slot_buffer_is_numerically_correct(rotate):
    N = TILE * BLOCKS * 4
    kernel = tilelang.compile(_manual_slots(N, rotate), out_idx=-1, target="ascend")
    a = torch.randn(N, device="npu")
    b = torch.randn(N, device="npu")
    torch.testing.assert_close(kernel(a, b), a + b, atol=1e-5, rtol=1e-5)
