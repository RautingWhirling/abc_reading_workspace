from __future__ import annotations

import pandas as pd

from .models import FeatureBuildResult, NodeFeature, NodeScore, ProductContext, ScoreResult, ScoreSummary


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, value))


class Scorer:
    def score(self, feature_result: FeatureBuildResult) -> ScoreResult:
        risk_limit = self._risk_limit(feature_result.event.constraints.risk_level)
        risk_penalty_factor = self._risk_penalty_factor(feature_result.event.constraints.risk_level)

        node_scores: list[NodeScore] = []
        for feature in feature_result.node_features:
            risk_flags, risk_score = self._compute_risk(feature)
            raw_dispatch_score = _clip_score(
                0.35 * feature.influence_score
                + 0.25 * feature.diffusion_score
                + 0.25 * feature.topic_match_score
                + 0.15 * feature.stability_score
            )
            final_score = _clip_score(raw_dispatch_score * (1 - risk_penalty_factor * risk_score))
            risk_level = self._risk_level(risk_score)
            manual_review_required = (
                risk_score >= 0.75
                or (feature_result.event.constraints.risk_level == "low" and risk_score > 0.35)
            )
            eligible = (
                risk_score <= risk_limit
                and feature.stability_score >= 0.20
                and (feature.topic_match_score > 0.0 or feature.influence_score >= 0.45)
            )
            priority_tier = self._priority_tier(final_score)
            selection_reasons = self._selection_reasons(
                feature=feature,
                risk_flags=risk_flags,
                final_score=final_score,
            )

            node_scores.append(
                NodeScore(
                    **feature.model_dump(),
                    raw_dispatch_score=round(raw_dispatch_score, 6),
                    risk_score=round(risk_score, 6),
                    risk_level=risk_level,
                    final_score=round(final_score, 6),
                    eligible=eligible,
                    manual_review_required=manual_review_required,
                    priority_tier=priority_tier,
                    risk_flags=risk_flags,
                    selection_reasons=selection_reasons,
                )
            )

        node_scores.sort(
            key=lambda item: (
                item.eligible,
                item.final_score,
                item.topic_match_score,
                item.influence_score,
                -item.risk_score,
            ),
            reverse=True,
        )

        summary = ScoreSummary(
            event_id=feature_result.event.event_id,
            event_type=feature_result.event.event_type,
            node_count=len(node_scores),
            eligible_count=sum(1 for item in node_scores if item.eligible),
            manual_review_count=sum(1 for item in node_scores if item.manual_review_required),
            high_priority_count=sum(1 for item in node_scores if item.priority_tier == "high"),
            avg_final_score=round(
                sum(item.final_score for item in node_scores) / max(len(node_scores), 1),
                6,
            ),
            avg_risk_score=round(
                sum(item.risk_score for item in node_scores) / max(len(node_scores), 1),
                6,
            ),
        )
        return ScoreResult(
            event=feature_result.event,
            product_context=feature_result.product_context,
            summary=summary,
            node_scores=node_scores,
        )

    def to_frame(self, score_result: ScoreResult) -> pd.DataFrame:
        return pd.DataFrame(item.model_dump() for item in score_result.node_scores)

    def _compute_risk(self, feature: NodeFeature) -> tuple[list[str], float]:
        total_interactions = feature.received_interaction_count + feature.made_interaction_count
        self_loop_ratio = feature.self_interaction_count / total_interactions if total_interactions else 0.0

        risk_components: list[float] = []
        risk_flags: list[str] = []

        profile_sparse_component = (1 - feature.profile_completeness_score) * 0.35
        if profile_sparse_component > 0.10:
            risk_flags.append("profile_sparse")
        risk_components.append(profile_sparse_component)

        if feature.neighbor_count == 0:
            risk_flags.append("isolated_node")
            risk_components.append(0.20)

        if self_loop_ratio > 0.10:
            risk_flags.append("self_interaction_high")
            risk_components.append(min(self_loop_ratio, 1.0) * 0.35)

        if feature.topic_match_score == 0.0:
            risk_flags.append("topic_mismatch")
            risk_components.append(0.18)
        elif feature.topic_match_score < 0.10:
            risk_flags.append("topic_match_weak")
            risk_components.append(0.08)

        if total_interactions > 0 and feature.mutual_neighbor_ratio < 0.05 and feature.diffusion_score > 0.40:
            risk_flags.append("interaction_concentration")
            risk_components.append(0.12)

        if not feature.has_description and feature.interest_count == 0:
            risk_flags.append("profile_context_missing")
            risk_components.append(0.15)

        risk_score = _clip_score(sum(risk_components))
        deduped_flags: list[str] = []
        for flag in risk_flags:
            if flag not in deduped_flags:
                deduped_flags.append(flag)
        return deduped_flags, risk_score

    def _selection_reasons(
        self,
        feature: NodeFeature,
        risk_flags: list[str],
        final_score: float,
    ) -> list[str]:
        reasons: list[str] = []
        if feature.influence_score >= 0.60:
            reasons.append("high_influence")
        if feature.diffusion_score >= 0.45:
            reasons.append("strong_diffusion")
        if feature.topic_match_score >= 0.20:
            reasons.append("good_topic_match")
        elif feature.topic_match_score > 0.0:
            reasons.append("some_topic_match")
        if feature.stability_score >= 0.70:
            reasons.append("stable_profile")
        if feature.influencer_flag:
            reasons.append("known_influencer")
        reasons.append(f"role_hint={feature.role_hint}")
        reasons.append(f"final_score={final_score:.3f}")
        if risk_flags:
            reasons.append(f"risk_flags={','.join(risk_flags)}")
        return reasons

    def _risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.75:
            return "high"
        if risk_score >= 0.40:
            return "medium"
        return "low"

    def _priority_tier(self, final_score: float) -> str:
        if final_score >= 0.55:
            return "high"
        if final_score >= 0.35:
            return "medium"
        return "low"

    def _risk_limit(self, risk_level: str) -> float:
        mapping = {
            "low": 0.35,
            "medium": 0.55,
            "high": 0.75,
        }
        return mapping.get(risk_level, 0.55)

    def _risk_penalty_factor(self, risk_level: str) -> float:
        mapping = {
            "low": 0.65,
            "medium": 0.50,
            "high": 0.35,
        }
        return mapping.get(risk_level, 0.50)
