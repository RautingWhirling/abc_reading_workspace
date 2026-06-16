from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import StrategyNodePlan, StrategyResult
from .pipeline import PipelineArtifacts, StrategyPipeline

TRACE_PREVIEW_LIMIT = 25


def load_hot_events(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict):
        events = [payload]
    else:
        raise ValueError("Hot event eval input must be a JSON object or a JSON array.")

    normalized_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each hot event item must be a JSON object.")
        normalized_events.append(event)
    return normalized_events


def select_hot_event(
    events: list[dict[str, Any]],
    event_id: str | None = None,
) -> dict[str, Any]:
    if not events:
        raise ValueError("Hot event eval input is empty.")
    if event_id is None:
        return events[0]

    for event in events:
        if str(event.get("event_id", "")) == event_id:
            return event
    raise ValueError(f"Hot event not found: {event_id}")


def select_hot_events(
    events: list[dict[str, Any]],
    *,
    event_id: str | None = None,
    event_limit: int | None = 10,
) -> list[dict[str, Any]]:
    if event_id is not None:
        return [select_hot_event(events, event_id=event_id)]
    if not events:
        raise ValueError("Hot event eval input is empty.")
    if event_limit is None:
        return events
    if event_limit < 1:
        raise ValueError("event_limit must be at least 1.")
    return events[:event_limit]


def hot_event_to_pipeline_payload(
    hot_event: dict[str, Any],
    *,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
) -> dict[str, Any]:
    event_id = str(hot_event.get("event_id") or "hot_event")
    event_title = str(hot_event.get("event_title") or event_id)
    event_summary = str(hot_event.get("event_summary") or "").strip()
    target = str(hot_event.get("target") or "扩大热点信息触达并引导理性讨论").strip()
    domain = str(hot_event.get("domain") or "general").strip() or "general"
    variants = _normalize_variants(hot_event.get("opinion_variants", []))

    variant_lines = "\n".join(
        f"{index}. {variant}" for index, variant in enumerate(variants, start=1)
    )
    event_description = "\n".join(
        item
        for item in (
            event_title,
            event_summary,
            f"热点领域: {domain}",
            f"传播目标: {target}",
            "任务约束: 输出影响力事件分发策略，不要按照商品推荐任务理解。",
            "热点叙述变体:",
            variant_lines,
        )
        if item
    )

    resolved_risk_level = risk_level or _infer_hot_event_risk_level(domain)
    return {
        "event_id": event_id,
        "product_name": "abc_reading",
        "event_title": event_title,
        "event_description": event_description,
        "target_goal": target,
        "target_audience": ["general_public"],
        "constraints": {
            "risk_level": resolved_risk_level,
            "max_selected_nodes": max_selected_nodes,
            "max_frequency_per_day": max_frequency_per_day,
            "campaign_window_hours": campaign_window_hours,
            "allowed_platforms": allowed_platforms or ["weibo_simulated"],
        },
    }


def run_hot_event_evaluation(
    *,
    workspace_root: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    event_id: str | None = None,
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    events = load_hot_events(input_path)
    hot_event = select_hot_event(events, event_id=event_id)
    return _run_hot_event_evaluation(
        workspace_root=workspace_root,
        output_dir=output_dir,
        hot_event=hot_event,
        profile_limit=profile_limit,
        max_selected_nodes=max_selected_nodes,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
        use_llm=use_llm,
        llm_client=llm_client,
        trace_dir=trace_dir,
    )


def run_hot_event_evaluations(
    *,
    workspace_root: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    event_id: str | None = None,
    event_limit: int | None = 10,
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    events = load_hot_events(input_path)
    selected_events = select_hot_events(
        events,
        event_id=event_id,
        event_limit=event_limit,
    )

    results: list[tuple[Path, dict[str, Any]]] = []
    for hot_event in selected_events:
        output_path, output_payload = _run_hot_event_evaluation(
            workspace_root=workspace_root,
            output_dir=output_dir,
            hot_event=hot_event,
            profile_limit=profile_limit,
            max_selected_nodes=max_selected_nodes,
            risk_level=risk_level,
            campaign_window_hours=campaign_window_hours,
            max_frequency_per_day=max_frequency_per_day,
            allowed_platforms=allowed_platforms,
            use_llm=use_llm,
            llm_client=llm_client,
            trace_dir=trace_dir,
        )
        results.append((output_path, output_payload))

    return results


def build_eval_output(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    workspace_root: str | Path | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    selected_nodes = strategy_result.selected_nodes
    selected_ids = [node.user_id for node in selected_nodes]
    platform = strategy_result.strategy.platform_plan.primary_platform
    target = str(
        hot_event.get("target") or strategy_result.event.target_goal or "扩大热点信息触达并引导理性讨论"
    ).strip()
    variants = _normalize_variants(hot_event.get("opinion_variants", []))

    llm_overrides, _llm_meta = _build_llm_node_texts(
        hot_event=hot_event,
        strategy_result=strategy_result,
        platform=platform,
        target=target,
        workspace_root=workspace_root,
        use_llm=use_llm,
        llm_client=llm_client,
    )

    payload: dict[str, Any] = {
        "事件名称": str(hot_event.get("event_title") or strategy_result.event.event_title),
        "输出格式版本": "action_schema_v2",
        "选取数字人id组": selected_ids,
        "策略概览": {
            "主平台": platform,
            "传播目标": target,
            "目标圈层": strategy_result.strategy.target_object,
            "传播窗口": strategy_result.strategy.time_plan,
            "主选节点数": len(selected_nodes),
        },
    }

    for node in selected_nodes:
        fallback_content = _fallback_content_bundle(
            hot_event=hot_event,
            strategy_result=strategy_result,
            node=node,
            target=target,
            variants=variants,
            selected_ids=selected_ids,
        )
        override_content = llm_overrides.get(node.user_id, {})
        audience_profile = override_content.get("audience_profile") or fallback_content["audience_profile"]
        interaction_focus = (
            override_content.get("audience_interaction_strategy")
            or fallback_content["audience_interaction_strategy"]
        )
        coordination_focus = (
            override_content.get("cross_digital_human_strategy")
            or fallback_content["cross_digital_human_strategy"]
        )
        topic_labels = _topic_labels(node, strategy_result)
        role_bucket_ids = _role_bucket_ids(strategy_result.selected_nodes, node.user_id)

        payload[node.user_id] = {
            "节点角色": {
                "角色类型": node.selected_role,
                "执行阶段": node.dispatch_stage,
                "时间阶段": _stage_text(node),
                "执行优先级": node.dispatch_priority,
                "节点得分": round(node.final_score, 6),
            },
            "执行目标": {
                "传播目标": target,
                "目标群体标签": strategy_result.strategy.target_object,
                "目标群体画像": audience_profile,
                "主题关键词": topic_labels,
            },
            "内容发布动作": _build_content_publish_actions(
                node=node,
                platform=platform,
                post_content=override_content.get("post_content") or fallback_content["post_content"],
                topic_labels=topic_labels,
                role_bucket_ids=role_bucket_ids,
            ),
            "受众互动动作": _build_audience_action_plan(
                node=node,
                topic_labels=topic_labels,
                interaction_focus=interaction_focus,
                role_bucket_ids=role_bucket_ids,
            ),
            "数字人协同动作": _build_coordination_action_plan(
                node=node,
                coordination_focus=coordination_focus,
                role_bucket_ids=role_bucket_ids,
            ),
            "风险控制动作": _build_risk_action_plan(node=node),
            "补充说明": {
                "节点入选原因": node.rationale[:6],
                "互动关注点": interaction_focus,
                "协同关注点": coordination_focus,
            },
        }

    return payload


def _topic_labels(node: StrategyNodePlan, strategy_result: StrategyResult) -> list[str]:
    labels: list[str] = []
    for item in [
        *node.matched_keywords,
        *node.semantic_tags,
        *strategy_result.event.extracted_keywords[:3],
    ]:
        text = str(item).strip()
        if text and text not in labels:
            labels.append(text)
    return labels[:6]


def _role_bucket_ids(
    selected_nodes: list[StrategyNodePlan],
    current_user_id: str,
) -> dict[str, list[str]]:
    role_map = {
        "core_publish_node": [],
        "interaction_response_node": [],
        "amplification_node": [],
        "support_node": [],
    }
    for node in selected_nodes:
        if node.user_id == current_user_id:
            continue
        role_map.setdefault(node.selected_role, []).append(node.user_id)
    return role_map


def _make_action(
    *,
    action_type: str,
    action_name: str,
    execute_window: str,
    trigger_condition: str,
    target_object: str,
    parameters: dict[str, Any],
    target_result: str,
) -> dict[str, Any]:
    return {
        "动作类型": action_type,
        "动作名称": action_name,
        "执行窗口": execute_window,
        "触发条件": trigger_condition,
        "目标对象": target_object,
        "执行参数": parameters,
        "目标结果": target_result,
    }


def _build_content_publish_actions(
    *,
    node: StrategyNodePlan,
    platform: str,
    post_content: str,
    topic_labels: list[str],
    role_bucket_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    role_action_map = {
        "core_publish_node": ("publish_post", "发布主帖"),
        "interaction_response_node": ("publish_followup_post", "发布跟进帖"),
        "amplification_node": ("quote_repost", "转发并附评"),
        "support_node": ("publish_support_post", "发布补充帖"),
    }
    action_type, action_name = role_action_map.get(
        node.selected_role,
        ("publish_post", "发布帖子"),
    )

    reference_ids = role_bucket_ids.get("core_publish_node", [])
    trigger_condition = "进入对应时间阶段后立即执行"
    target_object = "公共信息流"
    if node.selected_role in {"interaction_response_node", "amplification_node", "support_node"} and reference_ids:
        trigger_condition = f"核心节点 {reference_ids[0]} 完成首发后 10-30 分钟内执行"
        target_object = f"围绕数字人 {reference_ids[0]} 的核心信息展开"

    return [
        _make_action(
            action_type=action_type,
            action_name=action_name,
            execute_window=_stage_text(node),
            trigger_condition=trigger_condition,
            target_object=target_object,
            parameters={
                "执行平台": platform,
                "执行频次": _frequency_text(node.frequency_per_day),
                "内容风格": node.suggested_content_style,
                "建议话题标签": topic_labels,
                "正文草案": post_content,
                "关联数字人id": reference_ids[:2],
            },
            target_result="完成节点自身的阶段性内容发布任务，并为后续互动或扩散提供锚点。",
        )
    ]


def _build_audience_action_plan(
    *,
    node: StrategyNodePlan,
    topic_labels: list[str],
    interaction_focus: str,
    role_bucket_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        _make_action(
            action_type="monitor_comments",
            action_name="监测评论",
            execute_window=_stage_text(node),
            trigger_condition="帖子发出后持续监测评论区反馈",
            target_object="评论区与转评区",
            parameters={
                "优先关键词": topic_labels[:4],
                "关注类型": ["高频疑问", "事实补充", "明显误读"],
            },
            target_result="识别需要优先回应的问题与高价值讨论线索。",
        )
    ]

    if node.selected_role == "core_publish_node":
        actions.extend(
            [
                _make_action(
                    action_type="like_comment",
                    action_name="点赞评论",
                    execute_window=_stage_text(node),
                    trigger_condition="出现理性提问或有效补充信息的评论时执行",
                    target_object="用户评论",
                    parameters={
                        "数量上限": 5,
                        "优先评论类型": ["理性提问", "事实补充"],
                    },
                    target_result="提高优质评论的可见度，稳定讨论方向。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="前 3 个高频问题出现后执行",
                    target_object="高频问题评论",
                    parameters={
                        "数量上限": 3,
                        "回复风格": "解释型",
                        "回复要点": topic_labels[:3],
                    },
                    target_result="快速建立统一口径，避免首轮误读扩散。",
                ),
            ]
        )
    elif node.selected_role == "interaction_response_node":
        target_core = (role_bucket_ids.get("core_publish_node") or ["-"])[0]
        actions.extend(
            [
                _make_action(
                    action_type="comment_on_post",
                    action_name="评论主帖",
                    execute_window=_stage_text(node),
                    trigger_condition="核心主帖发布后立即跟进",
                    target_object=f"数字人 {target_core} 的主帖",
                    parameters={
                        "评论数量": 1,
                        "评论目的": "补充常见问题入口或追问锚点",
                        "互动关注点": interaction_focus,
                    },
                    target_result="把评论区讨论引导到可承接的问题上。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="评论区出现高频疑问、误读或追问时执行",
                    target_object="主帖评论区",
                    parameters={
                        "数量上限": 5,
                        "回复风格": "答疑型",
                        "引用核心节点": target_core,
                    },
                    target_result="集中消化高频问题，降低核心节点的回复压力。",
                ),
            ]
        )
    elif node.selected_role == "amplification_node":
        actions.extend(
            [
                _make_action(
                    action_type="like_post",
                    action_name="点赞帖子",
                    execute_window=_stage_text(node),
                    trigger_condition="核心节点完成首发后执行",
                    target_object=f"数字人 {(role_bucket_ids.get('core_publish_node') or ['-'])[0]} 的主帖",
                    parameters={
                        "执行次数": 1,
                    },
                    target_result="建立扩散节点与核心信息之间的可见连接。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="二次扩散帖下出现转化为事实问题的评论时执行",
                    target_object="扩散帖评论区",
                    parameters={
                        "数量上限": 3,
                        "回复风格": "摘要型",
                        "回复要点": topic_labels[:2],
                    },
                    target_result="让扩散环节保持一致口径，不偏离核心叙事。",
                ),
            ]
        )
    else:
        actions.extend(
            [
                _make_action(
                    action_type="like_comment",
                    action_name="点赞评论",
                    execute_window=_stage_text(node),
                    trigger_condition="出现有代表性的长尾问题或补充观点时执行",
                    target_object="长尾评论",
                    parameters={
                        "数量上限": 4,
                        "优先评论类型": ["背景追问", "案例补充"],
                    },
                    target_result="扶持更深入的讨论线索，避免讨论过快收缩。",
                ),
                _make_action(
                    action_type="reply_comment",
                    action_name="回复评论",
                    execute_window=_stage_text(node),
                    trigger_condition="需要补充背景信息时执行",
                    target_object="长尾问题评论",
                    parameters={
                        "数量上限": 2,
                        "回复风格": "补充说明型",
                        "互动关注点": interaction_focus,
                    },
                    target_result="补全背景信息，提升讨论完整度。",
                ),
            ]
        )

    return actions


def _build_coordination_action_plan(
    *,
    node: StrategyNodePlan,
    coordination_focus: str,
    role_bucket_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    core_ids = role_bucket_ids.get("core_publish_node", [])
    interaction_ids = role_bucket_ids.get("interaction_response_node", [])
    amplification_ids = role_bucket_ids.get("amplification_node", [])
    support_ids = role_bucket_ids.get("support_node", [])

    if node.selected_role == "core_publish_node":
        return [
            _make_action(
                action_type="reply_agent_comment",
                action_name="回复数字人评论",
                execute_window=_stage_text(node),
                trigger_condition="互动节点在主帖下完成第一轮评论后执行",
                target_object=f"数字人 {','.join(interaction_ids[:2]) or '-'} 的评论",
                parameters={
                    "动作目的": "确认关键信息并锁定统一口径",
                    "涉及数字人id": interaction_ids[:2],
                },
                target_result="形成核心节点与互动节点之间的可见协同。",
            ),
            _make_action(
                action_type="like_post",
                action_name="点赞数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="扩散或支持节点发布有效补充内容后执行",
                target_object=f"数字人 {','.join((amplification_ids + support_ids)[:2]) or '-'} 的帖子",
                parameters={
                    "数量上限": 2,
                    "协同关注点": coordination_focus,
                },
                target_result="放大协同节点的有效补充内容。",
            ),
        ]

    if node.selected_role == "interaction_response_node":
        return [
            _make_action(
                action_type="comment_on_post",
                action_name="评论数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="核心主帖发布后 5-15 分钟内执行",
                target_object=f"数字人 {','.join(core_ids[:1]) or '-'} 的主帖",
                parameters={
                    "评论数量": 1,
                    "评论目的": "承接问答和补充追问入口",
                    "涉及数字人id": core_ids[:1],
                },
                target_result="把互动节点挂接到核心帖下的主要讨论流。",
            ),
            _make_action(
                action_type="like_post",
                action_name="点赞数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="支持节点发布补充信息后执行",
                target_object=f"数字人 {','.join(support_ids[:2]) or '-'} 的补充帖",
                parameters={
                    "数量上限": 2,
                    "协同关注点": coordination_focus,
                },
                target_result="提高补充帖的可见度并形成协同反馈。",
            ),
        ]

    if node.selected_role == "amplification_node":
        return [
            _make_action(
                action_type="like_post",
                action_name="点赞数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="核心节点完成首发后执行",
                target_object=f"数字人 {','.join(core_ids[:1]) or '-'} 的主帖",
                parameters={
                    "执行次数": 1,
                    "涉及数字人id": core_ids[:1],
                },
                target_result="建立扩散节点与核心节点的显式关联。",
            ),
            _make_action(
                action_type="comment_on_post",
                action_name="评论数字人帖子",
                execute_window=_stage_text(node),
                trigger_condition="支持节点或互动节点发布补充信息后执行",
                target_object=f"数字人 {','.join((interaction_ids + support_ids)[:2]) or '-'} 的帖子",
                parameters={
                    "评论数量": 1,
                    "评论目的": "提炼亮点并导流回核心信息",
                    "协同关注点": coordination_focus,
                },
                target_result="让扩散内容和补充内容形成互相导流。",
            ),
        ]

    return [
        _make_action(
            action_type="comment_on_post",
            action_name="评论数字人帖子",
            execute_window=_stage_text(node),
            trigger_condition="核心帖或扩散帖出现需要补充背景的节点时执行",
            target_object=f"数字人 {','.join((core_ids + amplification_ids)[:2]) or '-'} 的帖子",
            parameters={
                "评论数量": 1,
                "评论目的": "补充背景、案例或上下文",
                "协同关注点": coordination_focus,
            },
            target_result="让支持节点承担背景补完角色，而不是重复主信息。",
        ),
        _make_action(
            action_type="like_comment",
            action_name="点赞数字人评论",
            execute_window=_stage_text(node),
            trigger_condition="互动节点完成高质量答疑后执行",
            target_object=f"数字人 {','.join(interaction_ids[:2]) or '-'} 的评论",
            parameters={
                "数量上限": 2,
                "涉及数字人id": interaction_ids[:2],
            },
            target_result="提高高质量答疑评论的曝光，稳定讨论节奏。",
        ),
    ]


def _build_risk_action_plan(node: StrategyNodePlan) -> list[dict[str, Any]]:
    return [
        _make_action(
            action_type="monitor_sentiment",
            action_name="监测舆情反馈",
            execute_window=_stage_text(node),
            trigger_condition="节点动作执行后持续观察反馈",
            target_object="评论区、转评区、协同节点反馈",
            parameters={
                "重点风险": node.risk_flags or ["内容同质化", "误读扩散"],
                "人工复核": node.manual_review_required,
            },
            target_result="尽快发现误读、负向升级或异常扩散。",
        ),
        _make_action(
            action_type="pause_dispatch_if_needed",
            action_name="必要时暂停动作",
            execute_window=_stage_text(node),
            trigger_condition="出现明显误读升级、敏感争议或与主口径冲突时执行",
            target_object="当前节点后续发布与互动动作",
            parameters={
                "暂停条件": ["负向评论快速上升", "敏感话题偏离", "与核心口径冲突"],
                "后续处理": "切换为人工复核或仅保留事实回应",
            },
            target_result="避免错误扩散和协同失真。",
        ),
    ]


def _run_hot_event_evaluation(
    *,
    workspace_root: str | Path,
    output_dir: str | Path | None = None,
    hot_event: dict[str, Any],
    profile_limit: int | None = None,
    max_selected_nodes: int = 5,
    risk_level: str | None = None,
    campaign_window_hours: int = 24,
    max_frequency_per_day: int = 3,
    allowed_platforms: list[str] | None = None,
    use_llm: bool = True,
    llm_client: Any | None = None,
    trace_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace_root)
    client = llm_client
    if client is None and use_llm:
        client = OpenAICompatibleLLMClient.from_env_files(root)

    pipeline_payload = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=max_selected_nodes,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
    )

    pipeline = StrategyPipeline(product_name="abc_reading")
    artifacts = pipeline.run_with_artifacts(
        workspace_root=root,
        event_input=pipeline_payload,
        profile_limit=profile_limit,
        use_llm=use_llm,
        llm_client=client,
    )
    output_payload = build_eval_output(
        hot_event=hot_event,
        strategy_result=artifacts.strategy_result,
        workspace_root=root,
        use_llm=use_llm,
        llm_client=client,
    )

    target_dir = Path(output_dir) if output_dir is not None else root / "eval" / "output"
    output_path = target_dir / f"{artifacts.strategy_result.event.event_id}_strategy_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_pipeline_trace(
        workspace_root=root,
        trace_dir=trace_dir,
        hot_event=hot_event,
        pipeline_payload=pipeline_payload,
        artifacts=artifacts,
        final_output=output_payload,
        output_path=output_path,
    )
    return output_path, output_payload


def _build_llm_node_texts(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    platform: str,
    target: str,
    workspace_root: str | Path | None,
    use_llm: bool,
    llm_client: Any | None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "requested": bool(use_llm),
        "configured": False,
        "attempted": False,
        "used": False,
        "provider": None,
        "model": None,
        "base_url": None,
        "fallback_reason": None,
    }

    if not use_llm:
        metadata["fallback_reason"] = "llm_disabled"
        return {}, metadata

    eligible_nodes = [
        node for node in strategy_result.selected_nodes if _can_connect_llm(node)
    ]
    if not eligible_nodes:
        metadata["fallback_reason"] = "no_eligible_nodes"
        return {}, metadata

    client = llm_client
    if client is None and workspace_root is not None:
        client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
    if client is None:
        metadata["fallback_reason"] = "llm_not_configured"
        return {}, metadata

    if hasattr(client, "describe"):
        details = client.describe()
        metadata["provider"] = details.get("provider")
        metadata["model"] = details.get("model")
        metadata["base_url"] = details.get("base_url")
    else:
        metadata["provider"] = getattr(client, "provider", None)
        metadata["model"] = getattr(client, "model", None)
        metadata["base_url"] = getattr(client, "base_url", None)
    metadata["configured"] = True
    metadata["attempted"] = True

    system_prompt = (
        "你是影响力事件分发策略的内容生成助手。"
        "请仅为已选出的匿名数字人生成中文执行文案。"
        "必须返回严格 JSON，不要输出 Markdown。"
    )
    user_prompt = _llm_prompt(
        hot_event=hot_event,
        strategy_result=strategy_result,
        eligible_nodes=eligible_nodes,
        platform=platform,
        target=target,
    )

    try:
        response = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (LLMClientError, OSError, ValueError, TypeError) as exc:
        metadata["fallback_reason"] = f"llm_request_failed:{type(exc).__name__}"
        return {}, metadata

    llm_nodes = _extract_llm_node_mapping(
        response=response,
        expected_ids=[node.user_id for node in eligible_nodes],
    )
    if not llm_nodes:
        metadata["fallback_reason"] = "invalid_llm_response"
        return {}, metadata

    overrides: dict[str, dict[str, str]] = {}
    for node in eligible_nodes:
        raw_node = llm_nodes.get(node.user_id)
        if not isinstance(raw_node, dict):
            continue
        cleaned = _clean_llm_node_fields(raw_node)
        if cleaned:
            overrides[node.user_id] = cleaned

    if overrides:
        metadata["used"] = True
    else:
        metadata["fallback_reason"] = "empty_llm_content"
    return overrides, metadata


def _fallback_content_bundle(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    node: StrategyNodePlan,
    target: str,
    variants: list[str],
    selected_ids: list[str],
) -> dict[str, Any]:
    return {
        "post_content": _post_content_text(node=node, target=target, variants=variants),
        "audience_profile": _audience_profile_text(
            hot_event=hot_event,
            strategy_result=strategy_result,
            node=node,
        ),
        "audience_interaction_strategy": _audience_interaction_text(node),
        "cross_digital_human_ids": [item for item in selected_ids if item != node.user_id],
        "cross_digital_human_strategy": _cross_digital_human_strategy(
            node,
            strategy_result.selected_nodes,
        ),
    }


def _can_connect_llm(node: StrategyNodePlan) -> bool:
    return (
        not node.manual_review_required
        and node.risk_level != "high"
        and node.final_score >= 0.30
        and node.stability_score >= 0.20
    )


def _llm_prompt(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    eligible_nodes: list[StrategyNodePlan],
    platform: str,
    target: str,
) -> str:
    variants = _normalize_variants(hot_event.get("opinion_variants", []))
    selected_ids = [node.user_id for node in strategy_result.selected_nodes]
    nodes_payload = []
    for node in eligible_nodes:
        nodes_payload.append(
            {
                "user_id": node.user_id,
                "role": node.selected_role,
                "stage": _stage_text(node),
                "frequency": _frequency_text(node.frequency_per_day),
                "platform": platform,
                "other_digital_human_ids": [item for item in selected_ids if item != node.user_id],
                "recommended_action": node.recommended_action,
                "content_style_hint": node.suggested_content_style,
                "metrics": {
                    "final_score": node.final_score,
                    "influence_score": node.influence_score,
                    "diffusion_score": node.diffusion_score,
                    "topic_match_score": node.topic_match_score,
                    "stability_score": node.stability_score,
                },
                "selection_reasons": node.rationale,
            }
        )

    prompt_payload = {
        "task": "为每个候选数字人生成结构化分发文案。",
        "event": {
            "event_id": hot_event.get("event_id"),
            "domain": hot_event.get("domain"),
            "event_title": hot_event.get("event_title"),
            "event_summary": hot_event.get("event_summary"),
            "target": target,
            "opinion_variants": variants,
        },
        "constraints": [
            "保持客观克制，适合公共热点信息传播。",
            "不要编造未给出的外部事实或真实平台权限。",
            "每个字段使用 1 到 3 句中文自然语言。",
            "高粉丝量不能替代主题相关性。",
            "返回 nodes 对象时，key 使用 id+user_id，例如 id81584。",
        ],
        "eligible_nodes": nodes_payload,
        "required_json_schema": {
            "nodes": {
                "id81584": {
                    "post_content": "发帖内容",
                    "audience_profile": "目标群体画像",
                    "audience_interaction_strategy": "目标群体交互策略",
                    "cross_digital_human_strategy": "与其他数字人的互动策略",
                }
            }
        },
    }
    return json.dumps(prompt_payload, ensure_ascii=False, indent=2)


def _clean_llm_node_fields(raw_node: dict[str, Any]) -> dict[str, str]:
    field_aliases = {
        "post_content": ("post_content", "发帖内容"),
        "audience_profile": ("audience_profile", "目标群体画像"),
        "audience_interaction_strategy": (
            "audience_interaction_strategy",
            "目标群体交互策略",
        ),
        "cross_digital_human_strategy": (
            "cross_digital_human_strategy",
            "互动策略",
            "与其他数字人互动策略",
        ),
    }

    cleaned: dict[str, str] = {}
    for normalized_name, aliases in field_aliases.items():
        for alias in aliases:
            value = raw_node.get(alias)
            if isinstance(value, str) and value.strip():
                cleaned[normalized_name] = value.strip()
                break
    return cleaned


def _extract_llm_node_mapping(
    *,
    response: dict[str, Any],
    expected_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(response, dict):
        return {}

    for key in ("nodes", "candidates", "results", "items", "data"):
        normalized = _normalize_llm_container(response.get(key), expected_ids)
        if normalized:
            return normalized

    normalized = _normalize_llm_container(response, expected_ids)
    if normalized:
        return normalized

    for value in response.values():
        normalized = _normalize_llm_container(value, expected_ids)
        if normalized:
            return normalized
    return {}


def _normalize_llm_container(
    container: Any,
    expected_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if isinstance(container, list):
        normalized: dict[str, dict[str, Any]] = {}
        for item in container:
            if not isinstance(item, dict):
                continue
            user_id = str(
                item.get("user_id")
                or item.get("id")
                or item.get("node_id")
                or item.get("digital_human_id")
                or ""
            ).strip()
            if user_id:
                normalized[user_id] = item
        return normalized

    if not isinstance(container, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for expected_id in expected_ids:
        raw_node = (
            container.get(expected_id)
            or container.get(str(expected_id))
            or container.get(f"id{expected_id}")
            or container.get(f"digital_human_{expected_id}")
        )
        if isinstance(raw_node, dict):
            normalized[str(expected_id)] = raw_node
    if normalized:
        return normalized

    for key, value in container.items():
        if not isinstance(value, dict):
            continue
        candidate_id = str(key).strip().removeprefix("id").removeprefix("digital_human_")
        if candidate_id in expected_ids:
            normalized[candidate_id] = value
    return normalized


def _stage_text(node: StrategyNodePlan) -> str:
    stage_map = {
        "stage_1_launch": "启动期",
        "stage_2_engage": "互动期",
        "stage_2_support": "支持期",
        "stage_3_amplify": "扩散期",
    }
    stage_label = stage_map.get(node.dispatch_stage, node.dispatch_stage)
    if node.timing_window:
        return f"{stage_label}（{node.timing_window}）"
    return stage_label


def _frequency_text(frequency_per_day: int) -> str:
    return f"{frequency_per_day}/day"


def _post_content_text(
    *,
    node: StrategyNodePlan,
    target: str,
    variants: list[str],
) -> str:
    variant_hint = ""
    if variants:
        variant_index = max(node.selection_rank - 1, 0) % len(variants)
        variant_hint = f"参考叙述变体：{variants[variant_index]}"

    parts = [
        f"传播目标：{target}",
        f"内容风格：{node.suggested_content_style}",
        f"执行动作：{node.recommended_action}",
        variant_hint,
    ]
    return " ".join(item for item in parts if item)


def _audience_profile_text(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    node: StrategyNodePlan,
) -> str:
    target_objects = "、".join(strategy_result.strategy.target_object) or "泛公众"
    domain = str(hot_event.get("domain") or "general")
    return (
        f"重点面向 {target_objects}，优先覆盖关注 {domain} 议题、"
        "需要事件解释与阶段性信息更新的人群。"
        f"该节点 final_score={node.final_score:.3f}。"
    )


def _audience_interaction_text(node: StrategyNodePlan) -> str:
    if node.selected_role == "core_publish_node":
        return "优先承接首轮评论，围绕核心事实、影响范围和后续行动建议引导讨论。"
    if node.selected_role == "interaction_response_node":
        return "重点回复高频问题，筛选理性追问并做二次解释，必要时回引核心发布内容。"
    if node.selected_role == "amplification_node":
        return "优先转述核心信息，面向相邻圈层用户做摘要式扩散和补充互动。"
    return "补充背景信息，回应长尾问题，避免与其他数字人重复表达。"


def _cross_digital_human_strategy(
    node: StrategyNodePlan,
    selected_nodes: list[StrategyNodePlan],
) -> str:
    core_ids = [
        item.user_id for item in selected_nodes
        if item.selected_role == "core_publish_node" and item.user_id != node.user_id
    ]
    engage_ids = [
        item.user_id for item in selected_nodes
        if item.selected_role == "interaction_response_node" and item.user_id != node.user_id
    ]
    amplify_ids = [
        item.user_id for item in selected_nodes
        if item.selected_role == "amplification_node" and item.user_id != node.user_id
    ]

    if node.selected_role == "core_publish_node":
        return (
            f"发布首帖后，由互动节点 {','.join(engage_ids) or '-'} 承接评论，"
            f"由扩散节点 {','.join(amplify_ids) or '-'} 在后续阶段做转述放大。"
        )
    if node.selected_role == "interaction_response_node":
        return (
            f"优先评论核心发布节点 {','.join(core_ids) or '-'} 的主帖，"
            "集中回复评论区高频问题，并将讨论回收到统一口径。"
        )
    if node.selected_role == "amplification_node":
        return (
            f"在扩散期转发核心节点 {','.join(core_ids) or '-'} 的内容，"
            f"必要时补充互动节点 {','.join(engage_ids) or '-'} 的答疑信息。"
        )
    return "对其他主选数字人的关键信息做补充说明和长尾维护。"


def _normalize_variants(raw_variants: Any) -> list[str]:
    if not isinstance(raw_variants, list):
        return []
    return [str(item).strip() for item in raw_variants if str(item).strip()]


def _infer_hot_event_risk_level(domain: str) -> str:
    if domain in {"military", "politics", "international_relations", "cybersecurity"}:
        return "high"
    if domain in {"finance", "financial_market", "energy", "public_policy"}:
        return "medium"
    return "low"


def _write_pipeline_trace(
    *,
    workspace_root: Path,
    trace_dir: str | Path | None,
    hot_event: dict[str, Any],
    pipeline_payload: dict[str, Any],
    artifacts: PipelineArtifacts,
    final_output: dict[str, Any],
    output_path: Path,
) -> None:
    target_root = Path(trace_dir) if trace_dir is not None else workspace_root / "tests" / "pipeline_step_outputs"
    event_dir = target_root / artifacts.strategy_result.event.event_id
    event_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        event_dir / "00_hot_event_input.json",
        hot_event,
    )
    _write_json(
        event_dir / "01_pipeline_payload.json",
        pipeline_payload,
    )
    _write_json(
        event_dir / "02_event_parser_output.json",
        artifacts.event.model_dump(mode="json"),
    )
    _write_json(
        event_dir / "03_feature_builder_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.feature_result.summary.model_dump(mode="json"),
            "event_keywords": {
                "event_title": artifacts.event.event_title,
                "extracted_keywords": artifacts.event.extracted_keywords,
                "semantic_tags": artifacts.event.semantic_tags,
                "target_audience": artifacts.event.target_audience,
            },
            "llm_feature_used_count": sum(
                1 for node in artifacts.feature_result.node_features if node.llm_feature_used
            ),
            "preview_top_features": [
                node.model_dump(mode="json")
                for node in artifacts.feature_result.node_features[:TRACE_PREVIEW_LIMIT]
            ],
        },
    )
    _write_json(
        event_dir / "04_scorer_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.score_result.summary.model_dump(mode="json"),
            "eligible_count": sum(1 for node in artifacts.score_result.node_scores if node.eligible),
            "preview_top_scores": [
                node.model_dump(mode="json")
                for node in artifacts.score_result.node_scores[:TRACE_PREVIEW_LIMIT]
            ],
        },
    )
    _write_json(
        event_dir / "05_selector_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.selection_result.summary.model_dump(mode="json"),
            "selected_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.selection_result.selected_nodes
            ],
            "fallback_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.selection_result.fallback_nodes[:TRACE_PREVIEW_LIMIT]
            ],
        },
    )
    _write_json(
        event_dir / "06_strategy_generator_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": artifacts.strategy_result.summary.model_dump(mode="json"),
            "stage_plans": [
                item.model_dump(mode="json")
                for item in artifacts.strategy_result.stage_plans
            ],
            "selected_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.strategy_result.selected_nodes
            ],
            "fallback_nodes": [
                node.model_dump(mode="json")
                for node in artifacts.strategy_result.fallback_nodes[:TRACE_PREVIEW_LIMIT]
            ],
            "strategy": artifacts.strategy_result.strategy.model_dump(mode="json"),
        },
    )
    _write_json(
        event_dir / "07_final_output.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "final_output_path": str(output_path),
            "payload": final_output,
        },
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
