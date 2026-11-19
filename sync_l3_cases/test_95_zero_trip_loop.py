"""L3 95: 空循环 / 零 trip.

代表用例：testing.ascend.auto_schedule.test_sync_boundaries::test_acquire_dominates_possibly_empty_consumer_loop
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

@pytest.mark.parametrize("counter_mode", [False, True], ids=["iteration-clock", "counter-clock"])
def test_acquire_dominates_possibly_empty_consumer_loop(counter_mode):
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    c = tirx.decl_buffer((4, 64), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    epoch = tirx.decl_buffer((1,), "int32", name="epoch", scope="local.var")
    i, j, count = (tirx.Var(name, "int32") for name in ("i", "j", "count"))
    child = tirx.For(j, 0, count, tirx.ForKind.SERIAL, unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, 64])))
    tasks = [unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, 64])), unit(child, core=None)]
    annotations = {"multi_buffer_eligible": [ub.data], "tl.storage_epoch_guard_map": {ub.data: tirx.const(True, "bool")}}
    if counter_mode:
        annotations["tl.multi_buffer_counter_map"] = {ub.data: epoch}
        tasks.append(unit(tirx.BufferStore(epoch, epoch[0] + 1, [0])))
    owner = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(*tasks), annotations=annotations), core=None)
    body = seq(unit(tirx.BufferStore(epoch, 0, [0])), owner) if counter_mode else owner
    before = kernel(
        body, buffers=[ub] + ([epoch] if counter_mode else []), params=[a, c, count], annotations={"tl.buffer_versions_map": {ub.data: 2}}
    )
    after = transform.InsertSync()(before)
    waits = list(_flag_sites(after, "wait", "MTE2_MTE3"))
    assert waits
    # Even count=0 must consume the producer's token. An acquire inside j
    # would leave an unmatched notification and corrupt the next ring reuse.
    assert all(not any(isinstance(parent, tirx.For) and parent.loop_var.same_as(j) for parent in parents) for _, parents in waits)
    assert all(any(isinstance(parent, tirx.For) and parent.loop_var.same_as(i) for parent in parents) for _, parents in waits)
