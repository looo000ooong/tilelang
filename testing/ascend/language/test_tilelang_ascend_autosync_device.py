"""Device-level auto-sync checks: multi-buffer slots, short loops, manual slots."""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing

pytest.importorskip("torch_npu")

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


if __name__ == "__main__":
    tilelang.testing.main()
