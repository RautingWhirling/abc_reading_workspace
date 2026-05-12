from __future__ import annotations

import math
import re
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
from rapidfuzz import fuzz

from .data_loader import DataLoader
from .models import (
    EnrichedUserProfile,
    FeatureBuildResult,
    FeatureBuildSummary,
    NodeFeature,
    ParsedEvent,
    ProductContext,
    UserProfile,
)

_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = _NON_TEXT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, value))


class FeatureBuilder:
    def __init__(self, product_name: str = "abc_reading") -> None:
        self.product_name = product_name

    def build_from_loader(
        self,
        workspace_root: str | Path,
        event: ParsedEvent,
        profile_limit: int | None = None,
    ) -> FeatureBuildResult:
        loader = DataLoader(workspace_root, product_name=self.product_name)
        product_context = loader.load_product_context()
        profiles = loader.load_profiles(limit=profile_limit)
        try:
            enriched_profiles = loader.load_enriched_profiles(limit=profile_limit)
        except (JSONDecodeError, ValueError):
            enriched_profiles = {}
        try:
            source_user_ids = set(
                loader.load_interactions(limit_sources=None, limit_records_per_source=0).keys()
            )
        except (JSONDecodeError, ValueError):
            source_user_ids = set()
        return self.build_features(
            product_context=product_context,
            profiles=profiles,
            event=event,
            enriched_profiles=enriched_profiles,
            source_user_ids=source_user_ids,
        )

    def build_features(
        self,
        product_context: ProductContext,
        profiles: dict[str, UserProfile],
        event: ParsedEvent,
        enriched_profiles: dict[str, EnrichedUserProfile] | None = None,
        source_user_ids: set[str] | None = None,
    ) -> FeatureBuildResult:
        enriched_profiles = enriched_profiles or {}
        source_user_ids = source_user_ids or set()
        user_ids = list(profiles.keys())
        influencer_ids = set(product_context.influencer_ids)

        follower_max_log = self._max_log(
            profiles[user_id].user_followers for user_id in user_ids
        )
        friend_max_log = self._max_log(
            profiles[user_id].user_friends for user_id in user_ids
        )
        neighbor_max_log = self._max_log(
            enriched_profiles.get(user_id, EnrichedUserProfile(
                user_id=user_id,
                user_name=profiles[user_id].user_name,
                user_followers=profiles[user_id].user_followers,
                user_friends=profiles[user_id].user_friends,
                user_interests=profiles[user_id].user_interests,
                user_description=profiles[user_id].user_description,
            )).graph_attributes.neighbor_count
            for user_id in user_ids
        )
        received_max_log = self._max_log(
            enriched_profiles.get(user_id, EnrichedUserProfile(
                user_id=user_id,
                user_name=profiles[user_id].user_name,
                user_followers=profiles[user_id].user_followers,
                user_friends=profiles[user_id].user_friends,
                user_interests=profiles[user_id].user_interests,
                user_description=profiles[user_id].user_description,
            )).graph_attributes.received_interaction_count
            for user_id in user_ids
        )
        made_max_log = self._max_log(
            enriched_profiles.get(user_id, EnrichedUserProfile(
                user_id=user_id,
                user_name=profiles[user_id].user_name,
                user_followers=profiles[user_id].user_followers,
                user_friends=profiles[user_id].user_friends,
                user_interests=profiles[user_id].user_interests,
                user_description=profiles[user_id].user_description,
            )).graph_attributes.made_interaction_count
            for user_id in user_ids
        )

        node_features: list[NodeFeature] = []
        keywords = event.extracted_keywords + event.target_audience
        normalized_keywords = self._prepare_keywords(keywords)

        for user_id, profile in profiles.items():
            enriched = enriched_profiles.get(user_id)
            graph_attributes = enriched.graph_attributes if enriched is not None else None

            follower_score = self._log_score(profile.user_followers, follower_max_log)
            friend_score = self._log_score(profile.user_friends, friend_max_log)
            neighbor_count = graph_attributes.neighbor_count if graph_attributes else 0
            mutual_neighbor_count = graph_attributes.mutual_neighbor_count if graph_attributes else 0
            received_interaction_count = graph_attributes.received_interaction_count if graph_attributes else 0
            made_interaction_count = graph_attributes.made_interaction_count if graph_attributes else 0
            received_comment_count = graph_attributes.received_comment_count if graph_attributes else 0
            received_repost_count = graph_attributes.received_repost_count if graph_attributes else 0
            made_comment_count = graph_attributes.made_comment_count if graph_attributes else 0
            made_repost_count = graph_attributes.made_repost_count if graph_attributes else 0
            self_interaction_count = graph_attributes.self_interaction_count if graph_attributes else 0
            has_graph_signal = bool(
                graph_attributes
                and (
                    neighbor_count
                    or received_interaction_count
                    or made_interaction_count
                    or mutual_neighbor_count
                )
            )

            neighbor_score = self._log_score(neighbor_count, neighbor_max_log)
            received_score = self._log_score(received_interaction_count, received_max_log)
            made_score = self._log_score(made_interaction_count, made_max_log)

            total_interaction_count = received_interaction_count + made_interaction_count
            total_comment_count = received_comment_count + made_comment_count
            total_repost_count = received_repost_count + made_repost_count
            comment_ratio = total_comment_count / total_interaction_count if total_interaction_count else 0.0
            repost_ratio = total_repost_count / total_interaction_count if total_interaction_count else 0.0
            mutual_neighbor_ratio = mutual_neighbor_count / neighbor_count if neighbor_count else 0.0

            matched_keywords = self._match_keywords(
                profile=profile,
                normalized_keywords=normalized_keywords,
            )
            topic_match_score = len(matched_keywords) / len(normalized_keywords) if normalized_keywords else 0.0
            profile_completeness_score = self._profile_completeness_score(profile)
            stability_score = self._stability_score(
                profile_completeness_score=profile_completeness_score,
                total_interaction_count=total_interaction_count,
                self_interaction_count=self_interaction_count,
            )
            if has_graph_signal:
                influence_score = _clip_score(
                    0.45 * follower_score
                    + 0.30 * received_score
                    + 0.15 * neighbor_score
                    + 0.10 * float(user_id in influencer_ids)
                )
                diffusion_score = _clip_score(
                    0.35 * neighbor_score
                    + 0.25 * made_score
                    + 0.25 * mutual_neighbor_ratio
                    + 0.15 * repost_ratio
                )
                activity_score = _clip_score(
                    0.45 * received_score
                    + 0.35 * made_score
                    + 0.10 * comment_ratio
                    + 0.10 * friend_score
                )
            else:
                influence_score = _clip_score(
                    0.75 * follower_score
                    + 0.15 * friend_score
                    + 0.10 * float(user_id in influencer_ids)
                )
                diffusion_score = _clip_score(0.45 * friend_score + 0.35 * follower_score)
                activity_score = _clip_score(0.55 * friend_score + 0.25 * follower_score)
            feature_ready_score = _clip_score(
                0.35 * influence_score
                + 0.25 * diffusion_score
                + 0.25 * topic_match_score
                + 0.15 * stability_score
            )

            role_hint = self._infer_role_hint(
                influence_score=influence_score,
                diffusion_score=diffusion_score,
                activity_score=activity_score,
                topic_match_score=topic_match_score,
                comment_ratio=comment_ratio,
            )

            node_features.append(
                NodeFeature(
                    user_id=user_id,
                    user_name=profile.user_name,
                    event_id=event.event_id,
                    event_type=event.event_type,
                    role_hint=role_hint,
                    influencer_flag=user_id in influencer_ids,
                    is_interaction_source=user_id in source_user_ids,
                    follower_count=profile.user_followers,
                    friend_count=profile.user_friends,
                    interest_count=len(profile.user_interests),
                    has_description=bool(profile.user_description.strip()),
                    neighbor_count=neighbor_count,
                    mutual_neighbor_count=mutual_neighbor_count,
                    received_interaction_count=received_interaction_count,
                    made_interaction_count=made_interaction_count,
                    received_comment_count=received_comment_count,
                    received_repost_count=received_repost_count,
                    made_comment_count=made_comment_count,
                    made_repost_count=made_repost_count,
                    self_interaction_count=self_interaction_count,
                    comment_ratio=round(comment_ratio, 6),
                    repost_ratio=round(repost_ratio, 6),
                    mutual_neighbor_ratio=round(mutual_neighbor_ratio, 6),
                    profile_completeness_score=round(profile_completeness_score, 6),
                    topic_match_score=round(topic_match_score, 6),
                    influence_score=round(influence_score, 6),
                    diffusion_score=round(diffusion_score, 6),
                    activity_score=round(activity_score, 6),
                    stability_score=round(stability_score, 6),
                    feature_ready_score=round(feature_ready_score, 6),
                    keyword_hit_count=len(matched_keywords),
                    matched_keywords=matched_keywords,
                )
            )

        node_features.sort(
            key=lambda item: (
                item.feature_ready_score,
                item.topic_match_score,
                item.influence_score,
                item.follower_count,
            ),
            reverse=True,
        )

        matched_node_count = sum(1 for item in node_features if item.keyword_hit_count > 0)
        summary = FeatureBuildSummary(
            event_id=event.event_id,
            event_type=event.event_type,
            node_count=len(node_features),
            source_node_count=sum(1 for item in node_features if item.is_interaction_source),
            matched_node_count=matched_node_count,
            avg_feature_ready_score=round(
                sum(item.feature_ready_score for item in node_features) / max(len(node_features), 1),
                6,
            ),
            avg_topic_match_score=round(
                sum(item.topic_match_score for item in node_features) / max(len(node_features), 1),
                6,
            ),
        )
        return FeatureBuildResult(
            event=event,
            product_context=product_context,
            summary=summary,
            node_features=node_features,
        )

    def to_frame(self, result: FeatureBuildResult) -> pd.DataFrame:
        rows = [item.model_dump() for item in result.node_features]
        return pd.DataFrame(rows)

    def _prepare_keywords(self, keywords: list[str]) -> list[str]:
        prepared: list[str] = []
        for keyword in keywords:
            normalized = _normalize_text(keyword)
            if not normalized:
                continue
            if normalized not in prepared:
                prepared.append(normalized)
        return prepared

    def _match_keywords(self, profile: UserProfile, normalized_keywords: list[str]) -> list[str]:
        if not normalized_keywords:
            return []

        search_text = _normalize_text(
            " ".join([profile.user_description, *profile.user_interests])
        )
        if not search_text:
            return []

        matches: list[str] = []
        tokens = search_text.split()
        compact_text = search_text.replace(" ", "")
        for keyword in normalized_keywords:
            if keyword in compact_text:
                matches.append(keyword)
                continue
            if any(fuzz.partial_ratio(keyword, token) >= 90 for token in tokens):
                matches.append(keyword)
        return matches

    def _profile_completeness_score(self, profile: UserProfile) -> float:
        has_description = 1.0 if profile.user_description.strip() else 0.0
        interest_score = min(len(profile.user_interests) / 4, 1.0)
        follower_presence = 1.0 if profile.user_followers > 0 else 0.0
        return _clip_score((has_description + interest_score + follower_presence) / 3)

    def _stability_score(
        self,
        profile_completeness_score: float,
        total_interaction_count: int,
        self_interaction_count: int,
    ) -> float:
        self_loop_ratio = self_interaction_count / total_interaction_count if total_interaction_count else 0.0
        self_loop_penalty = min(self_loop_ratio, 1.0)
        return _clip_score(0.75 * profile_completeness_score + 0.25 * (1 - self_loop_penalty))

    def _infer_role_hint(
        self,
        influence_score: float,
        diffusion_score: float,
        activity_score: float,
        topic_match_score: float,
        comment_ratio: float,
    ) -> str:
        if influence_score >= 0.6 and topic_match_score >= 0.2:
            return "core_broadcast"
        if diffusion_score >= 0.45:
            return "amplification"
        if activity_score >= 0.35 or comment_ratio >= 0.6:
            return "interaction_response"
        return "general_support"

    def _max_log(self, values) -> float:
        max_value = max((math.log1p(max(0, int(value))) for value in values), default=0.0)
        return max(max_value, 1.0)

    def _log_score(self, value: int, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return _clip_score(math.log1p(max(0, int(value))) / denominator)
