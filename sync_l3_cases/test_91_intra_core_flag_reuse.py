"""L3 91: 同核同步 id 合法且可复用.

代表用例：testing.ascend.auto_schedule.test_flag_allocation::test_serial_protocols_fit_hardware_flag_budget
"""

import pytest

import tilelang.ascend.language as T

import tilelang.testing

from tilelang import tvm

from tilelang.ascend import transform as ascend_transform

from tvm import tirx

from testing.ascend._ir import calls, schedule, statements

def _make_serial_protocols(count):
    @T.prim_func
    def main(A: T.Tensor((4 * count * 64,), "float32"), C: T.Tensor((4 * count * 64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({ub: (2, "iteration")})
            for i in T.Pipelined(4, num_stages=2):
                for j in T.Unroll(count, explicit=True):
                    offset = (i * count + j) * 64
                    T.copy(A[offset : offset + 64], ub)
                    with T.SimdVF():
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value)
                    T.copy(ub, C[offset : offset + 64])

    return main

def _insert_sync(program, *, infer_layouts=False):
    before = schedule(program, through="ResolveCore", auto_schedule=False, infer_layouts=infer_layouts)
    return ascend_transform.InsertSync()(before)

def _events(mod, operation, hard_event):
    return [call.args[1] for call in calls(mod, "tl.ascend_" + operation + "_flag") if str(call.args[0].value) == hard_event]

@pytest.mark.parametrize("count", [5, 9], ids=["fits", "requires-reuse"])
def test_serial_protocols_fit_hardware_flag_budget(count):
    mod = _insert_sync(_make_serial_protocols(count))
    for event in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
        sets = _events(mod, "set", event)
        waits = _events(mod, "wait", event)
        assert sets and waits
        assert sorted(str(expr) for expr in sets) == sorted(str(expr) for expr in waits)
        analyzer = tvm.arith.Analyzer()
        assert all(analyzer.can_prove(expr >= 0) and analyzer.can_prove(expr < 8) for expr in [*sets, *waits])
