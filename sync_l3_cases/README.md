# 同步 L3 特性代表用例

每个文件是表格 81–101 行对应 L3 特性的一个完整可执行用例（源码已复制，不依赖原测试文件）。

| 行 | L3 特性 | 代表用例 |
|---|---|---|
| 81 | 同迭代 RAW/WAR/WAW 与最小同步 | `test_81_same_iteration_raw_war_waw.py`（原 `testing.ascend.auto_schedule.test_sync_dependencies::test_dma_row_writes_only_serialize_overlapping_regions`） |
| 82 | 跨迭代 loop-carried 依赖 | `test_82_loop_carried_dependency.py`（原 `testing.ascend.auto_schedule.test_sync_dependencies::test_ring_flags_use_the_data_clock`） |
| 83 | 同 pipe 同步 | `test_83_same_pipe_sync.py`（原 `testing.ascend.auto_schedule.test_sync_dependencies::test_dma_issue_order_is_not_completion_order`） |
| 84 | 跨核同步 | `test_84_cross_core_sync.py`（原 `testing.ascend.auto_schedule.test_schedule_integration::test_subcore_gemm_results`） |
| 85 | PerCoreTask 多核任务同步 | `test_85_per_core_task_sync.py`（原 `testing.ascend.auto_schedule.test_tasks::test_sync_set_is_migrated_into_each_per_core_candidate`） |
| 86 | 多缓冲流水数据正确性 | `test_86_multibuffer_data_correctness.py`（原 `testing.ascend.language.test_tilelang_ascend_autosync_device::test_multibuffer_versions_match_single_buffer`） |
| 87 | 流水首尾与短循环 | `test_87_pipeline_prologue_epilogue.py`（原 `testing.ascend.language.test_tilelang_ascend_autosync_device::test_short_pipeline_does_not_hang_or_overrun`） |
| 88 | 条件跳轮 / 动态 trip | `test_88_conditional_skip.py`（原 `testing.ascend.language.test_tilelang_ascend_autosync_device::test_conditional_skip_then_normal_launch_has_no_residual_token`） |
| 89 | 手工多缓冲正确性 | `test_89_manual_multibuffer.py`（原 `testing.ascend.language.test_tilelang_ascend_autosync_device::test_manual_three_slot_buffer_is_numerically_correct`） |
| 90 | 嵌套与 alias 多缓冲 | `test_90_nested_alias_multibuffer.py`（原 `testing.ascend.auto_schedule.test_buffer_ownership::test_nested_alias_accesses_have_one_complete_owner`） |
| 91 | 同核同步 id 合法且可复用 | `test_91_intra_core_flag_reuse.py`（原 `testing.ascend.auto_schedule.test_flag_allocation::test_serial_protocols_fit_hardware_flag_budget`） |
| 92 | 同核 id 稀疏/越界的归位与降级 | `test_92_flag_compaction_and_spill.py`（原 `testing.ascend.auto_schedule.test_flag_allocation::test_sparse_out_of_range_flags_are_compacted_without_spill`） |
| 93 | 动态 id | `test_93_dynamic_flag_id.py`（原 `testing.ascend.auto_schedule.test_sync_spec_gaps::test_three_version_reverse_ring_uses_modulo_three`） |
| 94 | 共享通道与版本数优化 | `test_94_shared_channel_version_optimization.py`（原 `testing.ascend.auto_schedule.test_sync_spec_gaps::test_three_slot_declaration_with_reverse_distance_one_collapses_to_one_id`） |
| 95 | 空循环 / 零 trip | `test_95_zero_trip_loop.py`（原 `testing.ascend.auto_schedule.test_sync_boundaries::test_acquire_dominates_possibly_empty_consumer_loop`） |
| 96 | 嵌套与父子 guard | `test_96_nested_parent_child_guard.py`（原 `testing.ascend.auto_schedule.test_sync_boundaries::test_narrow_child_handshake_cannot_replace_wider_parent_dependency`） |
| 97 | 条件分支切换 | `test_97_branch_switch.py`（原 `testing.ascend.auto_schedule.test_sync_boundaries::test_branch_switch_guarded_siblings_share_one_acquire`） |
| 98 | 单次迭代 | `test_98_single_iteration.py`（原 `testing.ascend.auto_schedule.test_flag_allocation::test_single_iteration_needs_no_reverse_flag`） |
| 99 | 手动同核 / 跨核同步 | `test_99_manual_sync.py`（原 `testing.ascend.language.test_tilelang_ascend_intrinsics::test_manual_sync_intrinsic_codegen`） |
| 100 | 关闭自动调度 | `test_100_disable_auto_schedule.py`（原 `testing.ascend.auto_schedule.test_schedule_integration::test_unscheduled_pipeline_rejects_placement_instead_of_dropping_it`） |
| 101 | UnitFlag 用户管理 | `test_101_unit_flag.py`（原 `testing.ascend.auto_schedule.test_core_resolution::test_unit_flag_ctrl_suppresses_same_core_fix_sync`） |

运行：

```bash
python -m pytest sync_l3_cases -q --rootdir=$(pwd)
```
