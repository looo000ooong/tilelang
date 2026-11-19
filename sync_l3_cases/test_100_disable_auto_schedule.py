"""L3 100: 关闭自动调度.

代表用例：testing.ascend.auto_schedule.test_schedule_integration::test_unscheduled_pipeline_rejects_placement_instead_of_dropping_it
"""

import pytest

import tilelang

import tilelang.testing

import tilelang.ascend.language as T

from tilelang import tvm

from tvm import tirx

from testing.ascend._ir import nodes, calls, allocated_buffer

def _placed_versioned_kernel(b_address, unlimit=False, rows=16):
    @T.prim_func
    def main(A: T.Tensor((4, rows, 16), "bfloat16"), B: T.Tensor((4, 16, 16), "bfloat16"), C: T.Tensor((4, rows, 16), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((rows, 16), "bfloat16")
            b = T.alloc_l1((16, 16), "bfloat16")
            a_view = T.view(a, shape=(rows, 16))
            aq = T.alloc_l0a((rows, 16), "bfloat16")
            bk = T.alloc_l0b((16, 16), "bfloat16")
            c = T.alloc_l0c((rows, 16), "float32")
            T.annotate_buffer_versions({a: 2, b: 2, aq: 2, bk: 2, c: 2})
            T.annotate_buffer_addresses({a: 0, a_view: 0, b: b_address, aq: 512, bk: 1024, c: 2048})
            if unlimit:
                T.annotate_unlimit_memory("shared.l1")
            for i in T.Pipelined(4, num_stages=2, annotations={"enable_offset": True}):
                with T.Stage(0), T.Task():
                    T.copy(A[i, :, :], a)
                    T.copy(B[i, :, :], b)
                with T.Stage(1):
                    T.copy(a_view, aq)
                    T.copy(b, bk)
                    T.gemm(aq, bk, c, transpose_B=True, clear_accum=True)
                    T.copy(c, C[i, :, :])

    return main

def test_unscheduled_pipeline_rejects_placement_instead_of_dropping_it():
    with (
        pytest.raises(tvm.error.InternalError, match="requires the Ascend scheduled pipeline"),
        tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}),
    ):
        tilelang.lower(_placed_versioned_kernel(262144), target="ascend")
