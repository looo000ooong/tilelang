"""L3 85: PerCoreTask 多核任务同步.

代表用例：testing.ascend.auto_schedule.test_tasks::test_sync_set_is_migrated_into_each_per_core_candidate
"""

import pytest

import tilelang.ascend.language as T

from tilelang import tvm

from tilelang.ascend import transform

from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq, statements

from testing.ascend.auto_schedule._scheduled_ir import unit

@pytest.mark.parametrize("conditional_padding", [False, True], ids=["copy", "copy-then-conditional-fill"])
def test_sync_set_is_migrated_into_each_per_core_candidate(conditional_padding):
    a = tirx.decl_buffer((16, 16), "float32", name="A")
    b = tirx.decl_buffer((16, 16), "float32", name="B")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((16, 16), "float32", name="l0", scope="shared.l0a")
    enabled = tirx.Var("enabled", "bool")
    needs_padding = tirx.Var("needs_padding", "bool")
    padding = tirx.IfThenElse(
        needs_padding,
        tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(l1, 2), tirx.const(0, "float32")])),
        None,
    )
    candidates = seq(
        *[
            tirx.IfThenElse(guard, unit(seq(copy(source, l1), padding) if conditional_padding else copy(source, l1), core=2).body, None)
            for source, guard in ((a, enabled), (b, tirx.Not(enabled)))
        ]
    )
    group = tirx.AttrStmt({"core_mask": tirx.IntImm("int64", 2)}, "tl.ascend_per_core_task", 1, candidates)
    before = kernel(seq(unit(group, core=None), unit(copy(l1, l0), core=2)), buffers=[l1, l0], params=[a, b, enabled, needs_padding])
    after = transform.InsertSync()(before)
    sets = [
        stmt.value
        for stmt, _ in statements(after)
        if isinstance(stmt, tirx.Evaluate)
        and isinstance(stmt.value, tirx.Call)
        and stmt.value.op.name == "tl.ascend_set_flag"
        and stmt.value.args[0].value == "MTE2_MTE1"
    ]
    waits = [call for call in calls(after, "tl.ascend_wait_flag") if call.args[0].value == "MTE2_MTE1"]
    assert len(sets) == 2 and len(waits) == 1
    guards = []
    for stmt, parents in statements(after):
        if isinstance(stmt, tirx.Evaluate) and any(stmt.value.same_as(event) for event in sets):
            (guard,) = [parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse)]
            guards.append(guard)
            sequence = next(parent for parent in reversed(parents) if isinstance(parent, tirx.SeqStmt))
            (candidate,) = [child for child in sequence.seq if calls(child, "tl.tileop.ascend_copy")]
            assert sequence.seq[0].same_as(candidate)
            assert len(calls(candidate, "tl.tileop.fill")) == int(conditional_padding)
    assert len(guards) == 2
    assert tvm.arith.Analyzer().can_prove(tirx.Or(*guards))
    assert all(tvm.arith.Analyzer().can_prove_equal(event.args[1], waits[0].args[1]) for event in sets)
