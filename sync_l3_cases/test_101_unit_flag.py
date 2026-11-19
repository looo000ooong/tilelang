"""L3 101: UnitFlag 用户管理.

代表用例：testing.ascend.auto_schedule.test_core_resolution::test_unit_flag_ctrl_suppresses_same_core_fix_sync
"""

import pytest

import tilelang.ascend.language as T

from tilelang import tvm

from tilelang.ascend import transform

from tvm import tirx

from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq

from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit

def test_unit_flag_ctrl_suppresses_same_core_fix_sync():
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((16, 16), "float32", name="out")

    def insert(unit_flag):
        annotations = {"unit_flag_ctrl": tirx.IntImm("int32", unit_flag)}
        fix = copy(accum, ub, annotations=annotations)
        consume = copy(ub, out)
        before = kernel(seq(unit(fix, core=1), unit(consume, core=1)), buffers=[accum, ub], params=[out])
        return transform.InsertSync()(before)

    def fix_events(mod, operation):
        return [call for call in calls(mod, "tl.ascend_" + operation + "_flag") if "FIX" in call.args[0].value]

    after0 = insert(0)
    after3 = insert(3)
    assert fix_events(after0, "set") and fix_events(after0, "wait")
    assert not fix_events(after3, "set")
    assert not fix_events(after3, "wait")
