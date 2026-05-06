from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProductContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    product_name: str
    domain: str = "unknown"
    ads: str = ""
    influencer_ids: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    user_name: str
    user_followers: int = 0
    user_friends: int = 0
    user_interests: list[str] = Field(default_factory=list)
    user_description: str = ""


class GraphAttributes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    neighbor_count: int = 0
    engaged_by_neighbor_count: int = 0
    engaged_to_neighbor_count: int = 0
    mutual_neighbor_count: int = 0
    self_interaction_count: int = 0
    self_comment_count: int = 0
    self_repost_count: int = 0
    received_interaction_count: int = 0
    received_comment_count: int = 0
    received_repost_count: int = 0
    made_interaction_count: int = 0
    made_comment_count: int = 0
    made_repost_count: int = 0
    isolated: bool = True


class NeighborRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    neighbor_id: str
    relation: Literal["engaged_by", "engaged_to", "mutual", "isolated"]
    received_comment_count: int = 0
    received_repost_count: int = 0
    made_comment_count: int = 0
    made_repost_count: int = 0
    received_interaction_count: int = 0
    made_interaction_count: int = 0
    total_interaction_count: int = 0


class EnrichedUserProfile(UserProfile):
    graph_attributes: GraphAttributes = Field(default_factory=GraphAttributes)
    neighbors: list[NeighborRelation] = Field(default_factory=list)


class InteractionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_user_id: str
    target_user_id: str
    interact_type: str
    text_raw: str = ""
    text_comment: str = ""


class DatasetSummary(BaseModel):
    product_name: str
    profile_count: int
    interaction_source_count: int
    interaction_record_count: int
    enriched_profile_count: int | None = None


class DatasetBundle(BaseModel):
    product_context: ProductContext
    profiles: dict[str, UserProfile]
    interactions: dict[str, list[InteractionRecord]]
    enriched_profiles: dict[str, EnrichedUserProfile] | None = None
    summary: DatasetSummary


class EventConstraints(BaseModel):
    risk_level: Literal["low", "medium", "high"] = "medium"
    max_selected_nodes: int = Field(default=10, ge=1, le=100)
    max_frequency_per_day: int = Field(default=3, ge=1, le=24)
    campaign_window_hours: int = Field(default=24, ge=1, le=168)
    allowed_platforms: list[str] = Field(default_factory=lambda: ["weibo_simulated"])


class ParsedEvent(BaseModel):
    event_id: str
    product_name: str = "abc_reading"
    event_title: str
    event_description: str
    target_goal: str
    event_type: str
    target_audience: list[str] = Field(default_factory=list)
    extracted_keywords: list[str] = Field(default_factory=list)
    constraints: EventConstraints = Field(default_factory=EventConstraints)
    parser_name: str = "rule_based_v1"
    reasoning: list[str] = Field(default_factory=list)


class NodeFeature(BaseModel):
    user_id: str
    user_name: str
    event_id: str
    event_type: str
    role_hint: str
    influencer_flag: bool = False
    is_interaction_source: bool = False
    follower_count: int = 0
    friend_count: int = 0
    interest_count: int = 0
    has_description: bool = False
    neighbor_count: int = 0
    mutual_neighbor_count: int = 0
    received_interaction_count: int = 0
    made_interaction_count: int = 0
    received_comment_count: int = 0
    received_repost_count: int = 0
    made_comment_count: int = 0
    made_repost_count: int = 0
    self_interaction_count: int = 0
    comment_ratio: float = 0.0
    repost_ratio: float = 0.0
    mutual_neighbor_ratio: float = 0.0
    profile_completeness_score: float = 0.0
    topic_match_score: float = 0.0
    influence_score: float = 0.0
    diffusion_score: float = 0.0
    activity_score: float = 0.0
    stability_score: float = 0.0
    feature_ready_score: float = 0.0
    keyword_hit_count: int = 0
    matched_keywords: list[str] = Field(default_factory=list)


class FeatureBuildSummary(BaseModel):
    event_id: str
    event_type: str
    node_count: int
    source_node_count: int
    matched_node_count: int
    avg_feature_ready_score: float
    avg_topic_match_score: float


class FeatureBuildResult(BaseModel):
    event: ParsedEvent
    product_context: ProductContext
    summary: FeatureBuildSummary
    node_features: list[NodeFeature]


class NodeScore(NodeFeature):
    raw_dispatch_score: float = 0.0
    risk_score: float = 0.0
    risk_level: Literal["low", "medium", "high"] = "low"
    final_score: float = 0.0
    eligible: bool = False
    manual_review_required: bool = False
    priority_tier: Literal["high", "medium", "low"] = "low"
    risk_flags: list[str] = Field(default_factory=list)
    selection_reasons: list[str] = Field(default_factory=list)


class ScoreSummary(BaseModel):
    event_id: str
    event_type: str
    node_count: int
    eligible_count: int
    manual_review_count: int
    high_priority_count: int
    avg_final_score: float
    avg_risk_score: float


class ScoreResult(BaseModel):
    event: ParsedEvent
    product_context: ProductContext
    summary: ScoreSummary
    node_scores: list[NodeScore]
