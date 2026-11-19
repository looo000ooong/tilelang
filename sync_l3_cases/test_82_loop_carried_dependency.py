"""L3 82: 跨迭代 loop-carried 依赖.

代表用例：testing.ascend.auto_schedule.test_sync_dependencies::test_ring_flags_use_the_data_clock
"""

import pytest

from tilelang import tvm

from tilelang.ascend import language as T, transform

from tilelang.layout import make_ascend_major_k_layout, make_ascend_major_mn_layout

from tvm import tirx

from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq, statements

from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit

def _events(mod, operation, pipe):
    return [call for call in calls(mod, "tl.ascend_" + operation + "_flag") if call.args[0].value == pipe]

@pytest.mark.parametrize("mode, owners, guarded", [("iteration", 1, False), ("counter", 1, True), ("counter", 2, True)])
def test_ring_flags_use_the_data_clock(mode, owners, guarded):
    before = copy_ring(mode=mode, owners=owners, guarded=guarded, prepared=True)
    after = transform.InsertSync()(before)
    clock = allocated_buffer(before, "epoch")[0] if mode == "counter" else nodes(before, tirx.For)[0].loop_var
    analyzer = tvm.arith.Analyzer()
    for pipe in ("MTE2_MTE3", "MTE3_MTE2"):
        for operation in ("set", "wait"):
            events = _events(after, operation, pipe)
            dynamic = [event.args[1] for event in events if not isinstance(event.args[1], tirx.IntImm)]
            assert len(dynamic) == owners
            for flag in dynamic:
                base = analyzer.simplify(flag - clock % 2)
                assert isinstance(base, tirx.IntImm), "The ring must advance with the storage clock"
            if operation == "set":
                released = dynamic
            else:
                assert all(analyzer.can_prove_equal(left, right) for left, right in zip(released, dynamic))
    # Ring reuse needs an initial token and a final drain for every slot. IDs
    # need only match each other, not a particular allocator numbering.
    initial = {int(event.args[1]) for event in _events(after, "set", "MTE3_MTE2") if isinstance(event.args[1], tirx.IntImm)}
    drained = {int(event.args[1]) for event in _events(after, "wait", "MTE3_MTE2") if isinstance(event.args[1], tirx.IntImm)}
    assert initial == drained and len(initial) == 2
    if guarded:
        guarded_events = []
        for stmt, parents in statements(after):
            if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
                continue
            if stmt.value.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or isinstance(stmt.value.args[1], tirx.IntImm):
                continue
            owner = next(parent for parent in reversed(parents) if isinstance(parent, tirx.For))
            assert any(
                isinstance(parent, tirx.IfThenElse) and analyzer.can_prove_equal(parent.condition, owner.loop_var % 2 == 0)
                for parent in parents
            )
            guarded_events.append(stmt)
        assert guarded_events
