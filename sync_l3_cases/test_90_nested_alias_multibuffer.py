"""L3 90: 嵌套与 alias 多缓冲.

代表用例：testing.ascend.auto_schedule.test_buffer_ownership::test_nested_alias_accesses_have_one_complete_owner
"""

import pytest

from tilelang.ascend import transform

from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, seq

from testing.ascend.auto_schedule._scheduled_ir import unit

def test_nested_alias_accesses_have_one_complete_owner():
    base = tirx.decl_buffer((4,), "int32", name="base", scope="shared.dyn")
    alias = tirx.decl_buffer((2, 2), "int32", name="view", data=base.data)
    out = tirx.decl_buffer((4,), "int32", name="out")
    outer, inner = tirx.Var("outer", "int32"), tirx.Var("inner", "int32")
    inner_loop = tirx.For(inner, 0, 2, tirx.ForKind.SERIAL, unit(tirx.BufferStore(out, alias[0, inner], [inner])))
    outer_loop = tirx.For(outer, 0, 4, tirx.ForKind.SERIAL, seq(unit(tirx.BufferStore(base, outer, [0])), unit(inner_loop, core=None)))
    before = kernel(unit(outer_loop, core=None), buffers=[base], params=[out])
    after = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop for loop in nodes(after, tirx.For) if base.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert len(owners) == 1 and owners[0].loop_var.same_as(outer)
