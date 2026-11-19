"""L3 81: 同迭代 RAW/WAR/WAW 与最小同步.

代表用例：testing.ascend.auto_schedule.test_sync_dependencies::test_dma_row_writes_only_serialize_overlapping_regions
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

@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint-rows", "same-row"])
def test_dma_row_writes_only_serialize_overlapping_regions(overlap):
    a = tirx.decl_buffer((2, 64), "float32", name="A")
    out = tirx.decl_buffer((2, 64), "float32", name="out")
    ub = tirx.decl_buffer((2, 64), "float32", name="ub", scope="shared.dyn")
    row = tirx.Var("row", "int32")
    load = copy(a, ub, src_indices=[row, 0], dst_indices=[0 if overlap else row, 0], src_shape=[1, 64], dst_shape=[1, 64])
    rows = tirx.For(row, 0, 2, tirx.ForKind.SERIAL, unit(load))
    before = kernel(seq(unit(rows, core=None), unit(copy(ub, out))), buffers=[ub], params=[a, out])
    after = transform.InsertSync()(before)
    barriers = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE2"]
    assert bool(barriers) == overlap
    # The whole-buffer consumer must still wait for the row loads to finish.
    releases = _events(after, "set", "MTE2_MTE3")
    waits = _events(after, "wait", "MTE2_MTE3")
    assert len(releases) == len(waits) == 1
    tvm.ir.assert_structural_equal(releases[0].args[1], waits[0].args[1])
