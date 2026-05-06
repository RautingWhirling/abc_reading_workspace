from __future__ import annotations

from collections import Counter

import pandas as pd

from .models import NodeScore, ScoreResult, SelectedNode, SelectionResult, SelectionSummary


class Selector:
    def select(self, score_result: ScoreResult) -> SelectionResult:
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

        for role in ("core_broadcast", "interaction_response", "amplification"):
            if len(selected_nodes) >= max_selected:
                break
            candidate = self._first_by_role(safe_pool, role, used_ids)
            if candidate is not None:
                selected_nodes.append(self._make_selected_node(candidate, len(selected_nodes) + 1, "primary"))
                used_ids.add(candidate.user_id)

        for candidate in safe_pool:
            if len(selected_nodes) >= max_selected:
                break
            if candidate.user_id in used_ids:
                continue
            selected_nodes.append(self._make_selected_node(candidate, len(selected_nodes) + 1, "primary"))
            used_ids.add(candidate.user_id)

        for candidate in review_pool:
            if len(selected_nodes) >= max_selected:
                break
            if candidate.user_id in used_ids:
                continue
            selected_nodes.append(self._make_selected_node(candidate, len(selected_nodes) + 1, "primary"))
            used_ids.add(candidate.user_id)

        fallback_nodes: list[SelectedNode] = []
        fallback_candidates = [node for node in [*safe_pool, *review_pool, *fallback_pool] if node.user_id not in used_ids]
        for index, candidate in enumerate(fallback_candidates[:10], start=1):
            fallback_nodes.append(self._make_selected_node(candidate, index, "fallback"))

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
    ) -> NodeScore | None:
        for node in pool:
            if node.role_hint == role_hint and node.user_id not in used_ids:
                return node
        return None

    def _make_selected_node(
        self,
        node: NodeScore,
        selection_rank: int,
        selection_bucket: str,
    ) -> SelectedNode:
        selected_role = self._selected_role(node.role_hint)
        dispatch_stage = self._dispatch_stage(node.role_hint)
        dispatch_priority = self._dispatch_priority(node.priority_tier)
        selection_reasons = list(node.selection_reasons)
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
