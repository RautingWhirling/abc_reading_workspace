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
        "选取数字人id组": selected_ids,
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

        payload[node.user_id] = {
            "时间阶段": _stage_text(node),
            "发帖频率": _frequency_text(node.frequency_per_day),
            "发帖平台": platform,
            "发帖内容": override_content.get("post_content") or fallback_content["post_content"],
            "目标受众": {
                "目标群体画像": override_content.get("audience_profile")
                or fallback_content["audience_profile"],
                "目标群体交互策略": override_content.get("audience_interaction_strategy")
                or fallback_content["audience_interaction_strategy"],
            },
            "与其他数字人互动策略": {
                "互动数字人id集合": fallback_content["cross_digital_human_ids"],
                "互动策略": override_content.get("cross_digital_human_strategy")
                or fallback_content["cross_digital_human_strategy"],
            },
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
