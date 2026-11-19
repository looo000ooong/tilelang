"""L3 98: 单次迭代.

代表用例：testing.ascend.auto_schedule.test_flag_allocation::test_single_iteration_needs_no_reverse_flag
"""

import pytest

import tilelang.ascend.language as T

import tilelang.testing

from tilelang import tvm

from tilelang.ascend import transform as ascend_transform

from tvm import tirx

from testing.ascend._ir import calls, schedule, statements

def _make_inner_loop_program(extent: int):
    tile = 64

    @T.prim_func
    def main(A: T.Tensor((4 * tile,), "float32"), C: T.Tensor((tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(extent):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
            T.copy(ub, C)

    return main

def _insert_sync(program, *, infer_layouts=False):
    before = schedule(program, through="ResolveCore", auto_schedule=False, infer_layouts=infer_layouts)
    return ascend_transform.InsertSync()(before)

def _events(mod, operation, hard_event):
    return [call.args[1] for call in calls(mod, "tl.ascend_" + operation + "_flag") if str(call.args[0].value) == hard_event]

@pytest.mark.parametrize("extent", [1, 4])
def test_single_iteration_needs_no_reverse_flag(extent):
    mod = _insert_sync(_make_inner_loop_program(extent))
    assert bool(_events(mod, "set", "V_MTE2")) == (extent > 1)
    assert bool(_events(mod, "wait", "V_MTE2")) == (extent > 1)
