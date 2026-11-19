"""L3 97: 条件分支切换.

代表用例：testing.ascend.auto_schedule.test_sync_boundaries::test_branch_switch_guarded_siblings_share_one_acquire
"""

import pytest

from tilelang import tvm

from tilelang.ascend import transform

from tvm import tirx

from testing.ascend._ir import copy, kernel, seq, statements

from testing.ascend.auto_schedule._scheduled_ir import unit

def _flag_sites(mod, operation, pipe):
    for stmt, parents in statements(mod):
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            if call.op.name == "tl.ascend_" + operation + "_flag" and call.args[0].value == pipe:
                yield call, parents

def test_branch_switch_guarded_siblings_share_one_acquire():
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    b = tirx.decl_buffer((4, 64), "float32", name="B")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    phase = tirx.Var("phase", "bool")

    # The i-th iteration reads ub; the next iteration overwrites it from one of
    # two mutually exclusive branches. The scheduled IR represents if/else as
    # sibling guarded tasks, and one acquire must dominate both write paths.
    read = unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64]))
    write_a = unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64]), guard=phase)
    write_b = unit(copy(b, ub, src_indices=[i, 0], src_shape=[1, 64]), guard=tirx.Not(phase))
    body = seq(read, write_a, write_b)
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None),
        buffers=[ub],
        params=[a, b, c],
    )
    after = transform.InsertSync()(before)

    ordered = list(statements(after))
    branch_positions = [index for index, (stmt, _) in enumerate(ordered) if isinstance(stmt, tirx.IfThenElse)]
    assert len(branch_positions) == 2, "both guarded branches must survive InsertSync"
    waits = []
    for call, parents in _flag_sites(after, "wait", "MTE3_MTE2"):
        if not any(isinstance(parent, tirx.For) and parent.loop_var.same_as(i) for parent in parents):
            continue
        position = next(index for index, (stmt, _) in enumerate(ordered) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(call))
        waits.append(position)
    assert waits, "the cross-iteration acquire must exist"
    assert all(position < min(branch_positions) for position in waits), (
        "the cross-iteration acquire must dominate both guarded write branches"
    )
