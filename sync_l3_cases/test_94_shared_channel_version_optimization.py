"""L3 94: 共享通道与版本数优化.

代表用例：testing.ascend.auto_schedule.test_sync_spec_gaps::test_three_slot_declaration_with_reverse_distance_one_collapses_to_one_id
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

def _manual_ring(versions, rotate):
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((versions, 64), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    slot = i % versions if rotate else 0
    body = seq(
        unit(
            copy(
                a,
                ub,
                src_indices=[i, 0],
                src_shape=[1, 64],
                dst_indices=[slot, 0],
                dst_shape=[1, 64],
            )
        ),
        unit(
            copy(
                ub,
                c,
                src_indices=[slot, 0],
                src_shape=[1, 64],
                dst_indices=[i, 0],
                dst_shape=[1, 64],
            )
        ),
    )
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body)
    return kernel(
        unit(loop, core=None),
        buffers=[ub],
        params=[a, c],
        annotations={"tl.manual_multi_buffer": {ub.data: versions}},
    )

def test_three_slot_declaration_with_reverse_distance_one_collapses_to_one_id():
    before = _manual_ring(versions=3, rotate=False)
    after = transform.InsertSync()(before)

    forward_sets = _events(after, "set", "MTE2_MTE3")
    forward_waits = _events(after, "wait", "MTE2_MTE3")
    assert forward_sets and forward_waits
    assert all(isinstance(call.args[1], tirx.IntImm) for call in forward_sets), [
        str(call.args[1]) for call in forward_sets
    ]
    assert all(isinstance(call.args[1], tirx.IntImm) for call in forward_waits), [
        str(call.args[1]) for call in forward_waits
    ]
    assert {int(call.args[1]) for call in forward_sets} == {
        int(call.args[1]) for call in forward_waits
    }
    # Three declared slots, but the loop reuses one slot: only one id is used.
    assert len({int(call.args[1]) for call in forward_sets}) == 1
