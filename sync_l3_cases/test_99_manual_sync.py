"""L3 99: 手动同核 / 跨核同步.

代表用例：testing.ascend.language.test_tilelang_ascend_intrinsics::test_manual_sync_intrinsic_codegen
"""

import pytest

import tilelang

import tilelang.ascend.language as T

import tilelang.testing

from tilelang.ascend.language import simd as ascend_simd

from tilelang.engine.lower import lower

from tvm import tirx

from tvm.script.ir_builder import IRBuilder

from tvm.tirx import Call

from testing.ascend._ir import calls

def test_manual_sync_intrinsic_codegen():
    @T.prim_func
    def kernel(A: T.Tensor((1,), "float32"), flag: T.int32):
        with T.Kernel(1):
            T.ascend_pipe_barrier("PIPE_ALL")
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            T.ascend_sync_inter_arrive("PIPE_FIX", 3)
            T.ascend_sync_inter_wait("PIPE_MTE3", flag)
            T.ascend_threadfence()
            T.ascend_cross_core_set_flag(0, "PIPE_MTE3", 8)
            T.ascend_cross_core_wait_flag(0, "PIPE_MTE3", 8)
            A[0] = T.float32(1)

    source = lower(kernel, target="ascend").kernel_source
    for instruction in (
        "asc_sync();",
        "asc_threadfence();",
        "asc_sync_notify(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));",
        "asc_sync_wait(PIPE_S, PIPE_MTE3, static_cast<event_t>(0));",
        "asc_sync_inter_arrive(PIPE_FIX, 3);",
        "asc_sync_inter_wait(PIPE_MTE3, flag);",
        "asc_sync_inter_arrive(PIPE_MTE3, 8);",
        "asc_sync_inter_wait(PIPE_MTE3, 8);",
    ):
        assert instruction in source
