from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_client import LLMClientError, OpenAICompatibleLLMClient
from .models import NodeScore, ScoreResult, SelectedNode, SelectionResult, SelectionSummary
from .prompts import build_selector_system_prompt, build_selector_user_prompt


class Selector:
    def select(
        self,
        score_result: ScoreResult,
        *,
        workspace_root: str | Path | None = None,
        use_llm: bool = False,
        llm_client: Any | None = None,
    ) -> SelectionResult:
        max_selected = score_result.event.constraints.max_selected_nodes

        safe_pool = [
            node for node in score_result.node_scores if node.eligible and not node.manual_review_required
        ]
        review_pool = [
            node for node in score_result.node_scores if node.eligible and node.manual_review_required
        ]
        fallback_pool = [
            node for node in score_result.node_scores if not node.eligible or node.manual_review_required
        ]

        selected_nodes: list[SelectedNode] = []
        used_ids: set[str] = set()
        llm_selection_plan = self._llm_selection_plan(
            score_result=score_result,
            safe_pool=safe_pool,
            max_selected=max_selected,
            workspace_root=workspace_root,
            use_llm=use_llm,
            llm_client=llm_client,
        )
        preferred_roles = self._preferred_roles(score_result, llm_selection_plan)
        safe_pool = self._reorder_pool_with_llm(safe_pool, llm_selection_plan)

        for role in preferred_roles:
            if len(selected_nodes) >= max_selected:
                break
            candidate = self._first_by_role(
                safe_pool,
                role,
                used_ids,
                llm_selection_plan=llm_selection_plan,
            )
            if candidate is not None:
                selected_nodes.append(
                    self._make_selected_node(
                        candidate,
                        len(selected_nodes) + 1,
                        "primary",
                        llm_selection_plan=llm_selection_plan,
                    )
                )
                used_ids.add(candidate.user_id)

        for candidate in safe_pool:
            if len(selected_nodes) >= max_selected:
                break
            if candidate.user_id in used_ids:
                continue
            selected_nodes.append(
                self._make_selected_node(
                    candidate,
                    len(selected_nodes) + 1,
                    "primary",
                    llm_selection_plan=llm_selection_plan,
                )
            )
            used_ids.add(candidate.user_id)

        for candidate in review_pool:
            if len(selected_nodes) >= max_selected:
                break
            if candidate.user_id in used_ids:
                continue
            selected_nodes.append(
                self._make_selected_node(
                    candidate,
                    len(selected_nodes) + 1,
                    "primary",
                    llm_selection_plan=llm_selection_plan,
                )
            )
            used_ids.add(candidate.user_id)

        fallback_nodes: list[SelectedNode] = []
        fallback_candidates = [node for node in [*safe_pool, *review_pool, *fallback_pool] if node.user_id not in used_ids]
        for index, candidate in enumerate(fallback_candidates[:10], start=1):
            fallback_nodes.append(
                self._make_selected_node(
                    candidate,
                    index,
                    "fallback",
                    llm_selection_plan=llm_selection_plan,
                )
            )

        role_distribution = Counter(node.selected_role for node in selected_nodes)
        stage_distribution = Counter(node.dispatch_stage for node in selected_nodes)
        summary = SelectionSummary(
            event_id=score_result.event.event_id,
            event_type=score_result.event.event_type,
            max_selected_nodes=max_selected,
            selected_count=len(selected_nodes),
            fallback_count=len(fallback_nodes),
            selected_role_distribution=dict(role_distribution),
            selected_stage_distribution=dict(stage_distribution),
            avg_selected_final_score=round(
                sum(node.final_score for node in selected_nodes) / max(len(selected_nodes), 1),
                6,
            ),
        )

        return SelectionResult(
            event=score_result.event,
            product_context=score_result.product_context,
            summary=summary,
            selected_nodes=selected_nodes,
            fallback_nodes=fallback_nodes,
        )

    def to_frame(self, selection_result: SelectionResult, bucket: str = "primary") -> pd.DataFrame:
        if bucket == "fallback":
            rows = [item.model_dump() for item in selection_result.fallback_nodes]
        else:
            rows = [item.model_dump() for item in selection_result.selected_nodes]
        return pd.DataFrame(rows, columns=list(SelectedNode.model_fields.keys()))

    def _first_by_role(
        self,
        pool: list[NodeScore],
        role_hint: str,
        used_ids: set[str],
        llm_selection_plan: dict[str, Any] | None = None,
    ) -> NodeScore | None:
        for node in pool:
            if self._resolved_role_hint(node, llm_selection_plan) == role_hint and node.user_id not in used_ids:
                return node
        return None

    def _make_selected_node(
        self,
        node: NodeScore,
        selection_rank: int,
        selection_bucket: str,
        llm_selection_plan: dict[str, Any] | None = None,
    ) -> SelectedNode:
        role_hint = self._resolved_role_hint(node, llm_selection_plan)
        selected_role = self._selected_role(role_hint)
        dispatch_stage = self._dispatch_stage(role_hint)
        dispatch_priority = self._dispatch_priority(node.priority_tier)
        selection_reasons = list(node.selection_reasons)
        llm_reasoning = self._llm_reasoning(node.user_id, llm_selection_plan)
        selection_reasons.extend(llm_reasoning)
        selection_reasons.append(f"selected_role={selected_role}")
        selection_reasons.append(f"dispatch_stage={dispatch_stage}")
        selection_reasons.append(f"selection_bucket={selection_bucket}")

        return SelectedNode(
            **node.model_dump(exclude={"selection_reasons"}),
            selection_reasons=selection_reasons,
            selection_rank=selection_rank,
            selected_role=selected_role,
            dispatch_stage=dispatch_stage,
            dispatch_priority=dispatch_priority,
            selection_bucket=selection_bucket,
        )

    def _selected_role(self, role_hint: str) -> str:
        mapping = {
            "core_broadcast": "core_publish_node",
            "interaction_response": "interaction_response_node",
            "amplification": "amplification_node",
            "general_support": "support_node",
        }
        return mapping.get(role_hint, "support_node")

    def _dispatch_stage(self, role_hint: str) -> str:
        mapping = {
            "core_broadcast": "stage_1_launch",
            "interaction_response": "stage_2_engage",
            "amplification": "stage_3_amplify",
            "general_support": "stage_2_support",
        }
        return mapping.get(role_hint, "stage_2_support")

    def _dispatch_priority(self, priority_tier: str) -> str:
        mapping = {
            "high": "p1",
            "medium": "p2",
            "low": "p3",
        }
        return mapping.get(priority_tier, "p3")

    def _preferred_roles(
        self,
        score_result: ScoreResult,
        llm_selection_plan: dict[str, Any] | None = None,
    ) -> tuple[str, ...]:
        role_map = {
            "core_publish_node": "core_broadcast",
            "interaction_response_node": "interaction_response",
            "amplification_node": "amplification",
            "support_node": "general_support",
        }
        if llm_selection_plan:
            ordered_roles = []
            raw_role_map = llm_selection_plan.get("role_map", {})
            order_map = llm_selection_plan.get("order_map", {})
            for user_id, _index in sorted(order_map.items(), key=lambda item: item[1]):
                recommended_role = raw_role_map.get(user_id)
                mapped_role = role_map.get(recommended_role)
                if mapped_role and mapped_role not in ordered_roles:
                    ordered_roles.append(mapped_role)
            if ordered_roles:
                return tuple(ordered_roles)

        preferred = [
            role_map[item]
            for item in score_result.event.target_roles
            if item in role_map
        ]
        if not preferred:
            preferred = ["core_broadcast", "interaction_response", "amplification"]
        return tuple(dict.fromkeys(preferred))

    def _llm_selection_plan(
        self,
        *,
        score_result: ScoreResult,
        safe_pool: list[NodeScore],
        max_selected: int,
        workspace_root: str | Path | None,
        use_llm: bool,
        llm_client: Any | None,
    ) -> dict[str, Any] | None:
        if not use_llm or not safe_pool:
            return None

        client = llm_client
        if client is None and workspace_root is not None:
            client = OpenAICompatibleLLMClient.from_env_files(workspace_root)
        if client is None:
            return None

        shortlist = safe_pool[: score_result.event.dispatch_preferences.rerank_top_k]
        shortlist_payload = [
            {
                "user_id": node.user_id,
                "user_name": node.user_name,
                "role_hint": node.role_hint,
                "final_score": node.final_score,
                "llm_feature_score": node.llm_feature_score,
                "topic_match_score": node.topic_match_score,
                "risk_level": node.risk_level,
                "semantic_tags": node.semantic_tags,
                "matched_keywords": node.matched_keywords,
                "selection_reasons": node.selection_reasons[:6],
            }
            for node in shortlist
        ]
        try:
            response = client.generate_json(
                system_prompt=build_selector_system_prompt(),
                user_prompt=build_selector_user_prompt(
                    event=score_result.event,
                    max_selected=max_selected,
                    shortlist=shortlist_payload,
                ),
            )
        except (LLMClientError, OSError, ValueError, TypeError):
            return None

        selected_order = response.get("selected_order")
        fallback_order = response.get("fallback_order")
        global_notes = response.get("global_notes")
        if not isinstance(selected_order, list):
            return None

        order_map: dict[str, int] = {}
        role_map: dict[str, str] = {}
        reasoning_map: dict[str, list[str]] = {}
        for index, item in enumerate(selected_order):
            if not isinstance(item, dict):
                continue
            user_id = str(item.get("user_id", "")).strip()
            if not user_id:
                continue
            order_map[user_id] = index
            recommended_role = str(item.get("recommended_role", "")).strip()
            if recommended_role in {"core_publish_node", "interaction_response_node", "amplification_node", "support_node"}:
                role_map[user_id] = recommended_role
            reasoning_map[user_id] = [
                str(reason).strip()
                for reason in item.get("reasoning", [])
                if str(reason).strip()
            ][:4]

        normalized_fallback = [
            str(item).strip()
            for item in fallback_order
            if str(item).strip()
        ] if isinstance(fallback_order, list) else []
        normalized_notes = [
            str(item).strip()
            for item in global_notes
            if str(item).strip()
        ] if isinstance(global_notes, list) else []
        if not order_map:
            return None
        return {
            "order_map": order_map,
            "role_map": role_map,
            "reasoning_map": reasoning_map,
            "fallback_order": normalized_fallback,
            "global_notes": normalized_notes,
        }

    def _reorder_pool_with_llm(
        self,
        pool: list[NodeScore],
        llm_selection_plan: dict[str, Any] | None,
    ) -> list[NodeScore]:
        if not llm_selection_plan:
            return pool
        order_map = llm_selection_plan.get("order_map", {})
        return sorted(
            pool,
            key=lambda node: (
                order_map.get(node.user_id, len(order_map) + 1000),
                -node.final_score,
            ),
        )

    def _resolved_role_hint(
        self,
        node: NodeScore,
        llm_selection_plan: dict[str, Any] | None,
    ) -> str:
        if not llm_selection_plan:
            return node.role_hint
        role_map = llm_selection_plan.get("role_map", {})
        recommended_role = role_map.get(node.user_id)
        reverse_map = {
            "core_publish_node": "core_broadcast",
            "interaction_response_node": "interaction_response",
            "amplification_node": "amplification",
            "support_node": "general_support",
        }
        return reverse_map.get(recommended_role, node.role_hint)

    def _llm_reasoning(
        self,
        user_id: str,
        llm_selection_plan: dict[str, Any] | None,
    ) -> list[str]:
        if not llm_selection_plan:
            return []
        reasoning_map = llm_selection_plan.get("reasoning_map", {})
        reasoning = reasoning_map.get(user_id, [])
        return [f"llm_selector={item}" for item in reasoning]
