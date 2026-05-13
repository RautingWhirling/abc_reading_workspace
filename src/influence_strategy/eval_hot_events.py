from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import StrategyNodePlan, StrategyResult
from .pipeline import StrategyPipeline


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
    target = str(hot_event.get("target") or "扩大热点信息触达并引导理性讨论。").strip()
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
            f"传播目标: {target}",
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
        "target_audience": ["general_public", domain],
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
    )


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
) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace_root)
    pipeline_payload = hot_event_to_pipeline_payload(
        hot_event,
        max_selected_nodes=max_selected_nodes,
        risk_level=risk_level,
        campaign_window_hours=campaign_window_hours,
        max_frequency_per_day=max_frequency_per_day,
        allowed_platforms=allowed_platforms,
    )

    strategy_result = StrategyPipeline(product_name="abc_reading").run(
        workspace_root=root,
        event_input=pipeline_payload,
        profile_limit=profile_limit,
    )
    output_payload = build_eval_output(
        hot_event=hot_event,
        strategy_result=strategy_result,
        workspace_root=root,
        use_llm=use_llm,
        llm_client=llm_client,
    )

    target_dir = Path(output_dir) if output_dir is not None else root / "eval" / "output"
    output_path = target_dir / f"{strategy_result.event.event_id}_strategy_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, output_payload


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
    target = str(hot_event.get("target") or strategy_result.event.target_goal or "扩大热点信息触达并引导理性讨论。")
    llm_texts = _build_llm_node_texts(
        hot_event=hot_event,
        strategy_result=strategy_result,
        platform=platform,
        target=target,
        workspace_root=workspace_root,
        use_llm=use_llm,
        llm_client=llm_client,
    )

    output: dict[str, Any] = {
        "事件名称": str(hot_event.get("event_title") or strategy_result.event.event_title),
        "选取数字人id组": selected_ids,
    }
    variants = _normalize_variants(hot_event.get("opinion_variants", []))
    for node in selected_nodes:
        generated = llm_texts.get(node.user_id, {})
        output[f"数字人id{node.user_id}"] = {
            "时间阶段": _stage_text(node),
            "发帖频率": f"每日 {node.frequency_per_day} 次",
            "发帖平台": platform,
            "发帖内容": generated.get("发帖内容") or _post_content_text(
                node=node,
                target=target,
                variants=variants,
            ),
            "目标受众": {
                "目标群体画像": generated.get("目标群体画像") or _audience_profile_text(
                    hot_event=hot_event,
                    strategy_result=strategy_result,
                    node=node,
                ),
                "目标群体交互策略": generated.get("目标群体交互策略") or _audience_interaction_text(node),
            },
            "与其他数字人互动策略": {
                "互动数字人id集合": [item for item in selected_ids if item != node.user_id],
                "互动策略": generated.get("互动策略") or _cross_digital_human_strategy(node, selected_nodes),
            },
        }
    return output


def _digital_human_info(node: StrategyNodePlan) -> dict[str, Any]:
    return {
        "user_id": node.user_id,
        "user_name": node.user_name,
        "selected_role": node.selected_role,
        "selection_bucket": node.selection_bucket,
        "selection_rank": node.selection_rank,
        "dispatch_stage": node.dispatch_stage,
        "dispatch_priority": node.dispatch_priority,
        "timing_window": node.timing_window,
        "frequency_per_day": node.frequency_per_day,
        "recommended_action": node.recommended_action,
        "suggested_content_style": node.suggested_content_style,
        "selection_explanation": node.rationale,
        "matched_keywords": node.matched_keywords,
        "risk_level": node.risk_level,
        "risk_flags": node.risk_flags,
        "manual_review_required": node.manual_review_required,
        "metrics": {
            "final_score": node.final_score,
            "influence_score": node.influence_score,
            "diffusion_score": node.diffusion_score,
            "topic_match_score": node.topic_match_score,
            "stability_score": node.stability_score,
            "follower_count": node.follower_count,
            "friend_count": node.friend_count,
            "neighbor_count": node.neighbor_count,
            "mutual_neighbor_count": node.mutual_neighbor_count,
            "received_interaction_count": node.received_interaction_count,
            "made_interaction_count": node.made_interaction_count,
        },
    }


def _build_llm_node_texts(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    platform: str,
    target: str,
    workspace_root: str | Path | None,
    use_llm: bool,
    llm_client: Any | None,
) -> dict[str, dict[str, str]]:
    fallback = _fallback_node_texts(
        hot_event=hot_event,
        strategy_result=strategy_result,
        target=target,
    )
    if not use_llm:
        return fallback

    eligible_nodes = [
        node for node in strategy_result.selected_nodes if _can_connect_llm(node)
    ]
    if not eligible_nodes:
        return fallback

    client = llm_client
    if client is None and workspace_root is not None:
        client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
    if client is None:
        return fallback

    system_prompt = (
        "你是影响力事件分发策略的内容生成器。"
        "你只为已经通过数据集筛选的匿名数字人节点生成中文执行文案。"
        "必须输出严格 JSON，不要输出 Markdown。"
        "不要编造真实身份、真实平台权限或未给出的外部事实。"
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
    except (LLMClientError, OSError, ValueError, TypeError):
        return fallback

    llm_nodes = response.get("nodes", response)
    if not isinstance(llm_nodes, dict):
        return fallback

    result = dict(fallback)
    for node in eligible_nodes:
        raw_node = (
            llm_nodes.get(node.user_id)
            or llm_nodes.get(str(node.user_id))
            or llm_nodes.get(f"数字人id{node.user_id}")
            or llm_nodes.get(f"数字人ID{node.user_id}")
        )
        if not isinstance(raw_node, dict):
            continue
        cleaned = _clean_llm_node_fields(raw_node)
        if cleaned:
            result[node.user_id] = {
                **result.get(node.user_id, {}),
                **cleaned,
            }
    return result


def _fallback_node_texts(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    target: str,
) -> dict[str, dict[str, str]]:
    variants = _normalize_variants(hot_event.get("opinion_variants", []))
    result: dict[str, dict[str, str]] = {}
    for node in strategy_result.selected_nodes:
        result[node.user_id] = {
            "发帖内容": _post_content_text(node=node, target=target, variants=variants),
            "目标群体画像": _audience_profile_text(
                hot_event=hot_event,
                strategy_result=strategy_result,
                node=node,
            ),
            "目标群体交互策略": _audience_interaction_text(node),
            "互动策略": _cross_digital_human_strategy(node, strategy_result.selected_nodes),
        }
    return result


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
                "frequency": f"每日 {node.frequency_per_day} 次",
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
                    "follower_count": node.follower_count,
                    "friend_count": node.friend_count,
                    "neighbor_count": node.neighbor_count,
                    "received_interaction_count": node.received_interaction_count,
                    "made_interaction_count": node.made_interaction_count,
                },
                "selection_reasons": node.rationale,
            }
        )

    required_schema = {
        "nodes": {
            "数字人ID": {
                "发帖内容": "围绕事件目标生成的具体发帖内容或内容执行说明",
                "目标群体画像": "说明针对哪些人群",
                "目标群体交互策略": "说明@哪些粉丝、互动哪些评论、如何回复",
                "互动策略": "说明如何与其他数字人点赞、评论、转发或错峰配合",
            }
        }
    }
    prompt_payload = {
        "任务": "为每个 eligible node 生成输出 JSON 中四个文案字段，替换规则模板内容。",
        "事件": {
            "event_id": hot_event.get("event_id"),
            "domain": hot_event.get("domain"),
            "event_title": hot_event.get("event_title"),
            "event_summary": hot_event.get("event_summary"),
            "target": target,
            "opinion_variants": variants,
        },
        "全局约束": [
            "保持客观、克制、适合热点公共信息传播。",
            "不要写具体发布时间，只使用给定时间阶段。",
            "不要承诺真实平台执行能力。",
            "每个字段使用中文自然语言，长度控制在 1 到 3 句话。",
            "发帖内容必须结合事件标题、目标、叙述变体和该数字人的 role、recommended_action、content_style_hint 生成。",
            "每个数字人的四个文案字段都必须互相区分，不能复用同一句模板。",
            "返回 nodes 对象时，key 必须严格使用节点的 user_id 字符串。",
        ],
        "已评判可接入大模型的数字人节点": nodes_payload,
        "必须返回的 JSON 结构": required_schema,
    }
    return json.dumps(prompt_payload, ensure_ascii=False, indent=2)


def _clean_llm_node_fields(raw_node: dict[str, Any]) -> dict[str, str]:
    allowed_fields = ("发帖内容", "目标群体画像", "目标群体交互策略", "互动策略")
    result: dict[str, str] = {}
    for field in allowed_fields:
        value = raw_node.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()
    return result


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


def _post_content_text(
    *,
    node: StrategyNodePlan,
    target: str,
    variants: list[str],
) -> str:
    variant_hint = ""
    if variants:
        variant_index = max(node.selection_rank - 1, 0) % len(variants)
        variant_hint = f"参考叙述：{variants[variant_index]}"
    parts = [
        f"传播目的：{target}",
        f"内容风格：{node.suggested_content_style}",
        f"执行动作：{node.recommended_action}",
        variant_hint,
    ]
    return "；".join(item for item in parts if item)


def _audience_profile_text(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
    node: StrategyNodePlan,
) -> str:
    target_objects = "、".join(strategy_result.strategy.target_object) or "泛公众"
    domain = str(hot_event.get("domain") or "general")
    score_text = (
        f"该数字人 final_score={node.final_score:.3f}，"
        f"influence_score={node.influence_score:.3f}，"
        f"diffusion_score={node.diffusion_score:.3f}"
    )
    return f"面向{target_objects}，重点覆盖对 {domain} 话题敏感、关注公共信息和热点解释的人群；{score_text}。"


def _audience_interaction_text(node: StrategyNodePlan) -> str:
    if node.selected_role == "core_publish_node":
        return "优先承接首轮评论，@ 高相关粉丝和高质量评论用户，引导围绕事实、影响和应对建议展开讨论。"
    if node.selected_role == "interaction_response_node":
        return "重点回复评论区高频问题，筛选理性追问进行二次解释，必要时引用核心发布节点内容统一口径。"
    if node.selected_role == "amplification_node":
        return "优先转发核心信息，@ 相邻圈层粉丝，选择高赞或高信息量评论进行补充互动。"
    return "补充背景信息，回应长尾问题，避免与其他数字人重复表达。"


def _cross_digital_human_strategy(
    node: StrategyNodePlan,
    selected_nodes: list[StrategyNodePlan],
) -> str:
    core_ids = [item.user_id for item in selected_nodes if item.selected_role == "core_publish_node" and item.user_id != node.user_id]
    engage_ids = [item.user_id for item in selected_nodes if item.selected_role == "interaction_response_node" and item.user_id != node.user_id]
    amplify_ids = [item.user_id for item in selected_nodes if item.selected_role == "amplification_node" and item.user_id != node.user_id]

    if node.selected_role == "core_publish_node":
        targets = engage_ids + amplify_ids
        if targets:
            return f"发布首帖后，由互动节点 {','.join(engage_ids) or '无'} 承接评论，由扩散节点 {','.join(amplify_ids) or '无'} 分时段转发放大。"
        return "作为主发布节点独立完成首发，并等待其他节点补位互动。"
    if node.selected_role == "interaction_response_node":
        if core_ids:
            return f"优先评论和点赞核心发布节点 {','.join(core_ids)} 的帖子，集中回答评论区疑问并回引主帖。"
        return "围绕高频疑问进行评论互动，并把讨论收束到事实和行动建议。"
    if node.selected_role == "amplification_node":
        if core_ids:
            return f"在扩散期转发核心发布节点 {','.join(core_ids)} 的帖子，必要时点赞互动节点 {','.join(engage_ids) or '无'} 的答疑内容。"
        return "在扩散期转发高质量解释内容，并与其他扩散节点错峰发布。"
    return "点赞、评论或转发其他主选数字人的关键内容，承担补充说明和长尾维护。"


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
