from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import StrategyNodePlan, StrategyResult
from .pipeline import StrategyPipeline
from .reporting import render_eval_batch_markdown, render_eval_markdown, write_markdown

SCHEMA_VERSION = "eval_v2"


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

    if results:
        target_dir = Path(output_dir) if output_dir is not None else Path(workspace_root) / "eval" / "output"
        write_markdown(target_dir / "output.md", render_eval_batch_markdown(results))
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
    fallback_nodes = strategy_result.fallback_nodes
    selected_ids = [node.user_id for node in selected_nodes]
    platform = strategy_result.strategy.platform_plan.primary_platform
    target = str(hot_event.get("target") or strategy_result.event.target_goal or "扩大热点信息触达并引导理性讨论。").strip()

    llm_overrides, llm_meta = _build_llm_node_texts(
        hot_event=hot_event,
        strategy_result=strategy_result,
        platform=platform,
        target=target,
        workspace_root=workspace_root,
        use_llm=use_llm,
        llm_client=llm_client,
    )

    selected_digital_humans: list[dict[str, Any]] = []
    variants = _normalize_variants(hot_event.get("opinion_variants", []))
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
        content_output = {
            "post_content": override_content.get("post_content") or fallback_content["post_content"],
            "audience_profile": override_content.get("audience_profile") or fallback_content["audience_profile"],
            "audience_interaction_strategy": override_content.get("audience_interaction_strategy") or fallback_content["audience_interaction_strategy"],
            "cross_digital_human_ids": fallback_content["cross_digital_human_ids"],
            "cross_digital_human_strategy": override_content.get("cross_digital_human_strategy") or fallback_content["cross_digital_human_strategy"],
        }
        llm_generated_fields = sorted(key for key in override_content.keys() if key in {
            "post_content",
            "audience_profile",
            "audience_interaction_strategy",
            "cross_digital_human_strategy",
        })

        selected_digital_humans.append(
            {
                **_digital_human_summary(node=node, platform=platform),
                "content_output": content_output,
                "content_generation": {
                    "llm_generated_fields": llm_generated_fields,
                    "fallback_fields": sorted(
                        field
                        for field in (
                            "post_content",
                            "audience_profile",
                            "audience_interaction_strategy",
                            "cross_digital_human_strategy",
                        )
                        if field not in llm_generated_fields
                    ),
                },
            }
        )

    fallback_digital_humans = [
        _digital_human_summary(node=node, platform=platform)
        for node in fallback_nodes
    ]

    summary = {
        "event_id": strategy_result.event.event_id,
        "event_title": str(hot_event.get("event_title") or strategy_result.event.event_title),
        "event_type": strategy_result.event.event_type,
        "risk_level": strategy_result.strategy.risk_control.risk_level,
        "primary_platform": platform,
        "campaign_window_hours": strategy_result.event.constraints.campaign_window_hours,
        "selected_count": len(selected_nodes),
        "fallback_count": len(fallback_nodes),
        "selected_role_distribution": strategy_result.selection_summary.selected_role_distribution,
        "selected_digital_human_ids": selected_ids,
        "avg_selected_final_score": strategy_result.summary.avg_selected_final_score,
    }

    payload: dict[str, Any] = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": "hot_event_eval",
            "generator_mode": _generator_mode(llm_meta),
            "llm": llm_meta,
        },
        "event": {
            "event_id": str(hot_event.get("event_id") or strategy_result.event.event_id),
            "event_title": str(hot_event.get("event_title") or strategy_result.event.event_title),
            "event_summary": str(hot_event.get("event_summary") or "").strip(),
            "domain": str(hot_event.get("domain") or "general").strip() or "general",
            "target": target,
            "is_synthetic": bool(hot_event.get("is_synthetic", False)),
            "opinion_variants": variants,
        },
        "summary": summary,
        "stage_plans": [item.model_dump(mode="json") for item in strategy_result.stage_plans],
        "five_dimensions": {
            "target_object": strategy_result.strategy.target_object,
            "time_plan": strategy_result.strategy.time_plan,
            "frequency_plan": strategy_result.strategy.frequency_plan.model_dump(mode="json"),
            "platform_plan": strategy_result.strategy.platform_plan.model_dump(mode="json"),
            "content_plan": strategy_result.strategy.content_plan.model_dump(mode="json"),
        },
        "selected_digital_humans": selected_digital_humans,
        "fallback_digital_humans": fallback_digital_humans,
        "risk_control": strategy_result.strategy.risk_control.model_dump(mode="json"),
        "explainability": strategy_result.strategy.explainability,
    }
    return payload


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
    markdown_path = output_path.with_suffix(".md")
    output_payload["meta"]["output_files"] = {
        "json": str(output_path),
        "markdown": str(markdown_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(markdown_path, render_eval_markdown(output_payload))
    return output_path, output_payload


def _digital_human_summary(
    *,
    node: StrategyNodePlan,
    platform: str,
) -> dict[str, Any]:
    return {
        "selection_rank": node.selection_rank,
        "user_id": node.user_id,
        "user_name": node.user_name,
        "selected_role": node.selected_role,
        "dispatch_stage": node.dispatch_stage,
        "stage_text": _stage_text(node),
        "dispatch_priority": node.dispatch_priority,
        "timing_window": node.timing_window,
        "frequency_per_day": node.frequency_per_day,
        "frequency_text": _frequency_text(node.frequency_per_day),
        "platform": platform,
        "final_score": node.final_score,
        "risk_level": node.risk_level,
        "risk_flags": node.risk_flags,
        "manual_review_required": node.manual_review_required,
        "matched_keywords": node.matched_keywords,
        "recommended_action": node.recommended_action,
        "suggested_content_style": node.suggested_content_style,
        "selection_explanation": node.rationale,
        "metrics": {
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
        "请只为系统已经筛选出的匿名传播节点生成中文执行文案。"
        "必须返回严格 JSON，不要输出 Markdown。"
        "不要编造真实身份、平台权限或未给出的外部事实。"
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

    llm_nodes = response.get("nodes", response)
    if not isinstance(llm_nodes, dict):
        metadata["fallback_reason"] = "invalid_llm_response"
        return {}, metadata

    overrides: dict[str, dict[str, str]] = {}
    for node in eligible_nodes:
        raw_node = (
            llm_nodes.get(node.user_id)
            or llm_nodes.get(str(node.user_id))
            or llm_nodes.get(f"id{node.user_id}")
            or llm_nodes.get(f"digital_human_{node.user_id}")
        )
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
        "cross_digital_human_strategy": _cross_digital_human_strategy(node, strategy_result.selected_nodes),
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
                    "follower_count": node.follower_count,
                    "friend_count": node.friend_count,
                    "neighbor_count": node.neighbor_count,
                    "received_interaction_count": node.received_interaction_count,
                    "made_interaction_count": node.made_interaction_count,
                },
                "selection_reasons": node.rationale,
            }
        )

    prompt_payload = {
        "task": "请为每个 eligible node 生成四个中文文案字段，用于替换规则模板内容。",
        "event": {
            "event_id": hot_event.get("event_id"),
            "domain": hot_event.get("domain"),
            "event_title": hot_event.get("event_title"),
            "event_summary": hot_event.get("event_summary"),
            "target": target,
            "opinion_variants": variants,
        },
        "global_constraints": [
            "保持客观、克制，适合公共热点信息传播。",
            "不要编造未给出的外部事实或真实平台执行权限。",
            "每个字段使用中文自然语言，长度控制在 1 到 3 句话。",
            "四个字段必须相互区分，不要复用同一句模板。",
            "返回 nodes 对象时，key 必须使用 id + user_id，例如 id81584。",
        ],
        "eligible_nodes": nodes_payload,
        "required_json_schema": {
            "nodes": {
                "id81584": {
                    "post_content": "生成的发帖内容或执行文案",
                    "audience_profile": "目标群体画像说明",
                    "audience_interaction_strategy": "目标群体互动策略说明",
                    "cross_digital_human_strategy": "与其他数字人的互动协同策略",
                }
            }
        },
    }
    return json.dumps(prompt_payload, ensure_ascii=False, indent=2)


def _clean_llm_node_fields(raw_node: dict[str, Any]) -> dict[str, str]:
    field_aliases = {
        "post_content": ("post_content", "发帖内容"),
        "audience_profile": ("audience_profile", "目标群体画像"),
        "audience_interaction_strategy": ("audience_interaction_strategy", "目标群体互动策略"),
        "cross_digital_human_strategy": ("cross_digital_human_strategy", "互动策略", "与其他数字人互动策略"),
    }

    cleaned: dict[str, str] = {}
    for normalized_name, aliases in field_aliases.items():
        for alias in aliases:
            value = raw_node.get(alias)
            if isinstance(value, str) and value.strip():
                cleaned[normalized_name] = value.strip()
                break
    return cleaned


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
        variant_hint = f"参考叙述：{variants[variant_index]}"

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
        f"重点面向 {target_objects}，优先覆盖对 {domain} 议题敏感、"
        f"关注公共信息和热点解释的人群。该节点 final_score={node.final_score:.3f}，"
        f"influence_score={node.influence_score:.3f}，diffusion_score={node.diffusion_score:.3f}。"
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
            "集中回复评论区高频问题，并将讨论回收至统一口径。"
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


def _generator_mode(llm_meta: dict[str, Any]) -> str:
    if not llm_meta.get("requested"):
        return "rule_only"
    if llm_meta.get("used"):
        return "llm_enhanced"
    return "llm_fallback"
