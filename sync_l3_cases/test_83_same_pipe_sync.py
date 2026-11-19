"""L3 83: 同 pipe 同步.

代表用例：testing.ascend.auto_schedule.test_sync_dependencies::test_dma_issue_order_is_not_completion_order
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

def test_dma_issue_order_is_not_completion_order():
    a = tirx.decl_buffer((64,), "float32", name="A")
    c = tirx.decl_buffer((64,), "float32", name="C")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    # A DMA store must finish before MTE2 may overwrite its UB source.
    before = kernel(seq(unit(copy(ub, c)), unit(copy(a, ub))), buffers=[ub], params=[a, c])
    after = transform.InsertSync()(before)
    assert len(_events(after, "set", "MTE3_MTE2")) == 1
    assert len(_events(after, "wait", "MTE3_MTE2")) == 1
