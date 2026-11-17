"""IR-level coverage for auto-sync spec rows 83/89/93/94/98/101.

Prepared off-line; to be copied to
testing/ascend/auto_schedule/test_sync_spec_gaps.py once server 162 is back.
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


def _int_ids(mod, operation, pipe):
    return [int(call.args[1]) for call in _events(mod, operation, pipe) if isinstance(call.args[1], tirx.IntImm)]


# --- 94E: 3 declared slots but reverse distance 1 collapses to one id ---


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


def test_three_slot_rotating_ring_keeps_all_three_ids():
    before = _manual_ring(versions=3, rotate=True)
    after = transform.InsertSync()(before)
    dynamic = [
        call.args[1]
        for call in _events(after, "set", "MTE2_MTE3")
        if not isinstance(call.args[1], tirx.IntImm)
    ]
    assert len(dynamic) == 1
    clock = nodes(before, tirx.For)[0].loop_var
    base = tvm.arith.Analyzer().simplify(dynamic[0] - clock % 3)
    assert isinstance(base, tirx.IntImm)


# --- 93E: a dynamic id derived from the loop variable is bounded and emitted ---


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


# --- 83E: two Vector tasks on the same pipe need no flag ---


def test_back_to_back_vector_writes_need_no_flag():
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")

    def vector_store(value):
        return tirx.SBlock(
            [], [], [], "SIMD_VF", tirx.BufferStore(ub, tirx.const(value, "float32"), [0])
        )

    before = kernel(seq(unit(vector_store(1)), unit(vector_store(2))), buffers=[ub])
    after = transform.InsertSync()(before)
    assert not _events(after, "set", "V_V")
    assert not _events(after, "wait", "V_V")


# --- 89E: user-written modulo-3 slots keep working with automatic flags ---


def test_manual_three_slot_ring_keeps_user_indices_and_flags():
    a = tirx.decl_buffer((12, 64), "float32", name="A")
    c = tirx.decl_buffer((12, 64), "float32", name="C")
    ub = tirx.decl_buffer((3, 64), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    slot = i % 3
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
    before = kernel(
        unit(tirx.For(i, 0, 12, tirx.ForKind.SERIAL, body), core=None),
        buffers=[ub],
        params=[a, c],
    )
    after = transform.InsertSync()(before)
    assert _events(after, "set", "MTE2_MTE3")
    assert _events(after, "wait", "MTE2_MTE3")


# --- 98E: a single-trip loop still needs the forward handshake ---


def test_single_trip_loop_keeps_forward_handshake():
    a = tirx.decl_buffer((1, 64), "float32", name="A")
    c = tirx.decl_buffer((1, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = seq(
        unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64])),
        unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64])),
    )
    before = kernel(
        unit(tirx.For(i, 0, 1, tirx.ForKind.SERIAL, body), core=None),
        buffers=[ub],
        params=[a, c],
    )
    after = transform.InsertSync()(before)
    assert _events(after, "set", "MTE2_MTE3")
    assert _events(after, "wait", "MTE2_MTE3")
    # No next iteration writes the buffer, so the reverse release is gone.
    assert not _events(after, "set", "MTE3_MTE2")
    assert not _events(after, "wait", "MTE3_MTE2")


# --- 85F: scalar-only per-core candidates need no synchronization ---


def test_scalar_only_per_core_candidates_need_no_sync():
    sid = tirx.Var("sid", "int32")
    cells = [tirx.decl_buffer((1,), "float32", name=f"cell{index}", scope="local.var") for index in range(2)]
    candidates = seq(
        *[
            tirx.IfThenElse(
                sid == index,
                unit(tirx.BufferStore(cells[index], tirx.const(index, "float32"), [0]), core=0).body,
                None,
            )
            for index in range(2)
        ]
    )
    group = tirx.AttrStmt({"core_mask": 2}, "tl.ascend_per_core_task", 1, candidates)
    before = kernel(seq(unit(group, core=None)), buffers=cells, params=[sid])
    after = transform.InsertSync()(before)
    assert not calls(after, "tl.ascend_set_flag")
    assert not calls(after, "tl.ascend_wait_flag")


# --- 101F: an absent unit_flag_ctrl annotation is treated as zero ---


def test_absent_unit_flag_annotation_matches_zero():
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((16, 16), "float32", name="out")

    def insert(annotations):
        fix = copy(accum, ub, annotations=annotations)
        consume = copy(ub, out)
        before = kernel(
            seq(unit(fix, core=1), unit(consume, core=1)),
            buffers=[accum, ub],
            params=[out],
        )
        return transform.InsertSync()(before)

    def fix_events(mod, operation):
        return [
            call
            for call in calls(mod, "tl.ascend_" + operation + "_flag")
            if "FIX" in call.args[0].value
        ]

    absent = insert(None)
    zero = insert({"unit_flag_ctrl": tirx.IntImm("int32", 0)})
    for operation in ("set", "wait"):
        assert bool(fix_events(absent, operation)) == bool(fix_events(zero, operation))
