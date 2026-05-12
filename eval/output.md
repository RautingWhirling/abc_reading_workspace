# eval 输出格式说明

`run_eval.py` 会读取 `eval/hot_event_opinion_variants.json` 中的一条热点事件数据，并在 `eval/output/` 下生成：

```text
<event_id>_strategy_output.json
```

例如：

```text
eval/output/hot_event_001_strategy_output.json
```

## 顶层字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `generated_at` | string | 输出生成时间，UTC ISO 格式。 |
| `source_event` | object | 原始热点事件信息，包括事件 ID、领域、标题、摘要和 10 条叙述。 |
| `parsed_event` | object | 规则解析后的事件结构，用于后续分发策略生成。 |
| `summary` | object | 策略摘要，包括选中数量、备选数量、主平台、预计分发次数等。 |
| `five_dimensions` | object | 五个核心分发维度。 |
| `selected_digital_humans` | array | 被选中用于分发的数字人/传播节点信息。 |
| `fallback_digital_humans` | array | 备选数字人/传播节点信息。 |
| `risk_control` | object | 风险控制策略。 |
| `explainability` | array | 策略整体解释文本。 |
| `raw_strategy_result` | object | 底层完整策略结果，便于调试和二次开发。 |

## `source_event`

| 字段 | 说明 |
| --- | --- |
| `event_id` | 热点事件 ID。 |
| `domain` | 热点事件领域，例如 `military`、`technology`、`finance`。 |
| `event_title` | 热点事件标题。 |
| `event_summary` | 热点事件摘要。 |
| `opinion_variant_count` | 叙述数量，当前 eval 数据应为 `10`。 |
| `opinion_variants` | 十条不同叙述文本。 |

## `five_dimensions`

`five_dimensions` 固定包含 5 个维度：

| 字段 | 维度 | 说明 |
| --- | --- | --- |
| `distribution_object` | 分发对象 | 说明目标圈层、选中的数字人 ID、角色分布。 |
| `time_arrangement` | 时间安排 | 说明启动期、互动期、支持期、扩散期的时间窗口。 |
| `frequency_arrangement` | 频率安排 | 说明全局频次上限、不同角色每天建议触发次数、预计总分发次数。 |
| `platform_arrangement` | 平台安排 | 说明主平台、次级平台和模拟执行方式。 |
| `content_arrangement` | 内容安排 | 说明不同角色的内容风格和内容安全约束。 |

## `selected_digital_humans`

每个被选中的数字人包含：

| 字段 | 说明 |
| --- | --- |
| `user_id` | 匿名用户 ID，作为数字人/传播节点 ID。 |
| `user_name` | 数字人名称。 |
| `selected_role` | 分发角色，例如 `core_publish_node`、`interaction_response_node`、`amplification_node`、`support_node`。 |
| `selection_bucket` | 选择分组，主选为 `primary`。 |
| `selection_rank` | 选择排序。 |
| `dispatch_stage` | 分发阶段。 |
| `dispatch_priority` | 分发优先级。 |
| `timing_window` | 建议执行时间窗口。 |
| `frequency_per_day` | 每天建议触发次数。 |
| `recommended_action` | 建议动作。 |
| `suggested_content_style` | 建议内容风格。 |
| `selection_explanation` | 为什么选择该数字人的解释。 |
| `matched_keywords` | 与事件或受众命中的关键词。 |
| `risk_level` | 节点风险等级。 |
| `risk_flags` | 风险标签。 |
| `manual_review_required` | 是否需要人工复核。 |
| `metrics` | 评分和画像指标。 |

## `metrics`

`metrics` 用于解释数字人为什么被选中：

| 字段 | 说明 |
| --- | --- |
| `final_score` | 最终综合分。 |
| `influence_score` | 影响力分。 |
| `diffusion_score` | 扩散力分。 |
| `topic_match_score` | 话题匹配分。 |
| `stability_score` | 稳定性分。 |
| `follower_count` | 粉丝数。 |
| `friend_count` | 关注数。 |
| `neighbor_count` | 邻居节点数。 |
| `mutual_neighbor_count` | 双向邻居数。 |
| `received_interaction_count` | 被互动次数。 |
| `made_interaction_count` | 主动互动次数。 |

## `risk_control`

| 字段 | 说明 |
| --- | --- |
| `risk_level` | 事件风险等级。 |
| `review_required` | 是否需要整体人工复核。 |
| `manual_review_node_ids` | 需要人工复核的节点 ID。 |
| `fallback_trigger` | 启用备选节点或暂停扩散的触发条件。 |
| `notes` | 风险控制说明。 |
