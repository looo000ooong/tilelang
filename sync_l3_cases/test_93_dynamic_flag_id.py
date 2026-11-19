"""L3 93: 动态 id.

代表用例：testing.ascend.auto_schedule.test_sync_spec_gaps::test_three_version_reverse_ring_uses_modulo_three
"""

import pytest

from tilelang import tvm

from tilelang.ascend import transform

from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements

from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit

def _events(mod, operation, pipe):
    return [
        call
        for call in calls(mod, "tl.ascend_" + operation + "_flag")
        if call.args[0].value == pipe
    ]

def test_three_version_reverse_ring_uses_modulo_three():
    before = copy_ring(mode="iteration", versions=3, owners=1, prepared=True)
    after = transform.InsertSync()(before)

    dynamic = [
        call.args[1]
        for call in _events(after, "set", "MTE3_MTE2")
        if not isinstance(call.args[1], tirx.IntImm)
    ]
    assert len(dynamic) == 1
    clock = nodes(before, tirx.For)[0].loop_var
    base = tvm.arith.Analyzer().simplify(dynamic[0] - clock % 3)
    assert isinstance(base, tirx.IntImm)
