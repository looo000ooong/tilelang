"""L3 96: 嵌套与父子 guard.

代表用例：testing.ascend.auto_schedule.test_sync_boundaries::test_narrow_child_handshake_cannot_replace_wider_parent_dependency
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

def test_narrow_child_handshake_cannot_replace_wider_parent_dependency():
    a = tirx.decl_buffer((4, 128), "float32", name="A")
    c = tirx.decl_buffer((4, 128), "float32", name="C")
    ub = tirx.decl_buffer((128,), "float32", name="ub", scope="shared.dyn")
    p, q = tirx.Var("p", "bool"), tirx.Var("q", "bool")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")

    def load(offset, guard):
        return unit(copy(a, ub, src_indices=[i, offset], src_shape=[1, 64], dst_indices=[offset], dst_shape=[64]), guard=guard)

    def store(offset, guard):
        return unit(copy(ub, c, src_indices=[offset], src_shape=[64], dst_indices=[i, offset], dst_shape=[1, 64]), guard=guard)

    child = tirx.For(j, 0, 1, tirx.ForKind.SERIAL, seq(load(64, p), store(64, p)), annotations={"tl.storage_epoch_guard_map": {ub.data: p}})
    body = seq(load(0, q), unit(child, core=None), store(0, q))
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"tl.storage_epoch_guard_map": {ub.data: tirx.Or(p, q)}})
    before = kernel(unit(loop, core=None), buffers=[ub], params=[a, c, p, q])
    after = transform.InsertSync()(before)
    # Evaluate the structural guards for p=False,q=True: a parent handshake
    # must still execute even though all inner work (and its handshake) skips.
    for operation in ("set", "wait"):
        covering = []
        for call, parents in _flag_sites(after, operation, "MTE2_MTE3"):
            if any(isinstance(parent, tirx.For) and parent.loop_var.same_as(j) for parent in parents):
                continue
            guards = [parent.condition for parent in parents if isinstance(parent, tirx.IfThenElse)]
            active = tirx.all(*guards) if guards else tirx.const(True, "bool")
            active = tirx.stmt_functor.substitute(active, {p: tirx.const(False, "bool"), q: tirx.const(True, "bool")})
            if tvm.arith.Analyzer().can_prove(active):
                covering.append(call)
        assert covering, "The child-only ordering does not cover the parent access"
