from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import StrategyNodePlan, StrategyResult
from .pipeline import StrategyPipeline

FIVE_DIMENSION_KEYS = (
    "distribution_object",
    "time_arrangement",
    "frequency_arrangement",
    "platform_arrangement",
    "content_arrangement",
)


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
        "target_goal": "awareness",
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
) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace_root)
    events = load_hot_events(input_path)
    hot_event = select_hot_event(events, event_id=event_id)
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
    )

    target_dir = Path(output_dir) if output_dir is not None else root / "eval" / "output"
    output_path = target_dir / f"{strategy_result.event.event_id}_strategy_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path, output_payload


def build_eval_output(
    *,
    hot_event: dict[str, Any],
    strategy_result: StrategyResult,
) -> dict[str, Any]:
    selected_digital_humans = [
        _digital_human_info(node) for node in strategy_result.selected_nodes
    ]
    fallback_digital_humans = [
        _digital_human_info(node) for node in strategy_result.fallback_nodes
    ]

    five_dimensions = {
        "distribution_object": {
            "dimension_name": "分发对象",
            "target_object": strategy_result.strategy.target_object,
            "selected_digital_human_ids": [
                node["user_id"] for node in selected_digital_humans
            ],
            "role_distribution": strategy_result.selection_summary.selected_role_distribution,
        },
        "time_arrangement": {
            "dimension_name": "时间安排",
            "time_plan": strategy_result.strategy.time_plan,
            "stage_plans": [
                stage.model_dump(mode="json") for stage in strategy_result.stage_plans
            ],
        },
        "frequency_arrangement": {
            "dimension_name": "频率安排",
            "frequency_plan": strategy_result.strategy.frequency_plan.model_dump(mode="json"),
            "estimated_total_dispatches": strategy_result.summary.estimated_total_dispatches,
        },
        "platform_arrangement": {
            "dimension_name": "平台安排",
            "platform_plan": strategy_result.strategy.platform_plan.model_dump(mode="json"),
        },
        "content_arrangement": {
            "dimension_name": "内容安排",
            "content_plan": strategy_result.strategy.content_plan.model_dump(mode="json"),
            "content_guardrails": strategy_result.strategy.content_plan.general_guardrails,
        },
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_event": {
            "event_id": str(hot_event.get("event_id", "")),
            "domain": str(hot_event.get("domain", "")),
            "event_title": str(hot_event.get("event_title", "")),
            "event_summary": str(hot_event.get("event_summary", "")),
            "opinion_variant_count": len(_normalize_variants(hot_event.get("opinion_variants", []))),
            "opinion_variants": _normalize_variants(hot_event.get("opinion_variants", [])),
        },
        "parsed_event": strategy_result.event.model_dump(mode="json"),
        "summary": strategy_result.summary.model_dump(mode="json"),
        "five_dimensions": five_dimensions,
        "selected_digital_humans": selected_digital_humans,
        "fallback_digital_humans": fallback_digital_humans,
        "risk_control": strategy_result.strategy.risk_control.model_dump(mode="json"),
        "explainability": strategy_result.strategy.explainability,
        "raw_strategy_result": strategy_result.model_dump(mode="json"),
    }


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
