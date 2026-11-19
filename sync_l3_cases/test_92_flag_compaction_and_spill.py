"""L3 92: 同核 id 稀疏/越界的归位与降级.

代表用例：testing.ascend.auto_schedule.test_flag_allocation::test_sparse_out_of_range_flags_are_compacted_without_spill
"""

import pytest

import tilelang.ascend.language as T

import tilelang.testing

from tilelang import tvm

from tilelang.ascend import transform as ascend_transform

from tvm import tirx

from testing.ascend._ir import calls, schedule, statements

def _make_sparse_out_of_range_flag_program():
    @T.prim_func
    def main():
        with T.Kernel(1):
            for group_iter in T.serial(4):
                T.ascend_set_flag("V_MTE3", group_iter % 2)
                T.ascend_wait_flag("V_MTE3", group_iter % 2)
            for group_iter in T.serial(4):
                T.ascend_set_flag("V_MTE3", group_iter % 2 + 2)
                T.ascend_wait_flag("V_MTE3", group_iter % 2 + 2)
            T.ascend_set_flag("V_MTE3", 4)
            T.ascend_wait_flag("V_MTE3", 4)
            T.ascend_set_flag("V_MTE3", 6)
            T.ascend_wait_flag("V_MTE3", 6)
            T.ascend_set_flag("V_MTE3", 8)
            T.ascend_wait_flag("V_MTE3", 8)

    return main

def _event_ranges(mod, operation, hard_event):
    ranges = []
    op_name = "tl.ascend_" + operation + "_flag"
    for stmt, ancestors in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op) or call.op.name != op_name or str(call.args[0].value) != hard_event:
            continue
        analyzer = tvm.arith.Analyzer()
        for ancestor in ancestors:
            if isinstance(ancestor, tirx.For):
                analyzer.bind(ancestor.loop_var, tvm.ir.Range.from_min_extent(ancestor.min, ancestor.extent))
        bound = analyzer.const_int_bound(call.args[1])
        ranges.append((bound.min_value, bound.max_value))
    return sorted(ranges)

def test_sparse_out_of_range_flags_are_compacted_without_spill():
    mod = tvm.IRModule({"main": _make_sparse_out_of_range_flag_program()})
    expected_before = [(0, 1), (2, 3), (4, 4), (6, 6), (8, 8)]
    for operation in ("set", "wait"):
        assert _event_ranges(mod, operation, "V_MTE3") == expected_before

    rewritten = ascend_transform.RewriteFlagToBuf()(mod)
    assert not calls(rewritten, "tl.ascend_get_buf")
    assert not calls(rewritten, "tl.ascend_rls_buf")
    expected_after = [(0, 1), (2, 3), (4, 4), (5, 5), (6, 6)]
    for operation in ("set", "wait"):
        assert _event_ranges(rewritten, operation, "V_MTE3") == expected_after
