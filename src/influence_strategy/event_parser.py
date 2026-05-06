from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from rapidfuzz import fuzz

from .models import EventConstraints, ParsedEvent

_SPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)

EVENT_TYPE_RULES: dict[str, tuple[str, ...]] = {
    "public_opinion_response": ("争议", "投诉", "质疑", "回应", "澄清", "危机", "辟谣", "负面"),
    "activity_announcement": ("活动", "直播", "专场", "福利", "报名", "上线", "发布", "通知"),
    "english_learning_engagement": ("英语", "英文", "启蒙", "分级阅读", "单词", "口语"),
    "parent_child_reading": ("亲子", "家长", "阅读", "绘本", "家庭教育", "陪读"),
    "reading_habit_campaign": ("阅读计划", "阅读打卡", "书单", "读书", "阅读习惯", "阅读推广"),
}

GOAL_RULES: dict[str, tuple[str, ...]] = {
    "engagement": ("讨论", "互动", "评论", "参与", "交流"),
    "awareness": ("传播", "曝光", "扩散", "覆盖", "触达"),
    "response": ("回应", "澄清", "解释", "辟谣", "稳定"),
    "conversion": ("报名", "转化", "参与报名", "下载", "注册"),
}

AUDIENCE_RULES: dict[str, tuple[str, ...]] = {
    "parent_child": ("亲子", "家长", "育儿", "家庭教育"),
    "reading_interest": ("阅读", "绘本", "书单", "读书"),
    "english_learning": ("英语", "英文", "启蒙", "口语", "单词"),
    "education_practitioner": ("老师", "教育", "学校", "课堂"),
}

HIGH_RISK_KEYWORDS = ("危机", "投诉", "举报", "造假", "翻车", "维权", "道歉")
MEDIUM_RISK_KEYWORDS = ("争议", "质疑", "刷屏", "负面", "误解", "冲突")


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    cleaned = _NON_TEXT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", cleaned).strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class RuleBasedEventParser:
    def __init__(self, default_product_name: str = "abc_reading") -> None:
        self.default_product_name = default_product_name

    def parse(self, payload: str | dict[str, Any]) -> ParsedEvent:
        raw_event = self._normalize_payload(payload)
        merged_text = " ".join(
            [
                raw_event["event_title"],
                raw_event["event_description"],
                raw_event["target_goal"],
                " ".join(raw_event["target_audience"]),
            ]
        )
        normalized_text = _normalize_text(merged_text)

        event_type, event_reasons = self._infer_event_type(normalized_text)
        target_goal, goal_reasons = self._infer_target_goal(normalized_text, raw_event["target_goal"])
        target_audience, audience_reasons = self._infer_target_audience(normalized_text, raw_event["target_audience"])
        extracted_keywords = self._extract_keywords(normalized_text, raw_event["target_audience"])
        constraints = self._normalize_constraints(raw_event.get("constraints", {}), normalized_text)

        reasoning = _dedupe_preserve_order(
            event_reasons + goal_reasons + audience_reasons + [f"risk_level={constraints.risk_level}"]
        )
        return ParsedEvent(
            event_id=raw_event["event_id"],
            product_name=raw_event["product_name"],
            event_title=raw_event["event_title"],
            event_description=raw_event["event_description"],
            target_goal=target_goal,
            event_type=event_type,
            target_audience=target_audience,
            extracted_keywords=extracted_keywords,
            constraints=constraints,
            reasoning=reasoning,
        )

    def _normalize_payload(self, payload: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, str):
            title = payload.strip()[:24] or "unnamed_event"
            return {
                "event_id": self._make_event_id(),
                "product_name": self.default_product_name,
                "event_title": title,
                "event_description": payload.strip(),
                "target_goal": "",
                "target_audience": [],
                "constraints": {},
            }

        if not isinstance(payload, dict):
            raise TypeError("Event payload must be a string or a dictionary.")

        event_description = str(payload.get("event_description", "")).strip()
        event_title = str(payload.get("event_title") or event_description[:24] or "unnamed_event").strip()
        target_goal = str(payload.get("target_goal", "")).strip()
        target_audience = [str(item).strip() for item in payload.get("target_audience", []) if str(item).strip()]
        return {
            "event_id": str(payload.get("event_id") or self._make_event_id()),
            "product_name": str(payload.get("product_name") or self.default_product_name),
            "event_title": event_title,
            "event_description": event_description,
            "target_goal": target_goal,
            "target_audience": target_audience,
            "constraints": payload.get("constraints", {}),
        }

    def _infer_event_type(self, normalized_text: str) -> tuple[str, list[str]]:
        best_type = "general_influence_event"
        best_score = 0.0
        reasons: list[str] = []

        for event_type, keywords in EVENT_TYPE_RULES.items():
            score = self._keyword_score(normalized_text, keywords)
            if score > best_score:
                best_type = event_type
                best_score = score
                reasons = [f"event_type={event_type}", f"matched_keywords={','.join(self._matched_keywords(normalized_text, keywords))}"]

        if best_score == 0.0:
            reasons = ["event_type=general_influence_event", "matched_keywords=none"]
        return best_type, reasons

    def _infer_target_goal(self, normalized_text: str, explicit_goal: str) -> tuple[str, list[str]]:
        if explicit_goal:
            return explicit_goal, [f"target_goal=explicit:{explicit_goal}"]

        best_goal = "awareness"
        best_score = 0.0
        reasons: list[str] = []
        for goal, keywords in GOAL_RULES.items():
            score = self._keyword_score(normalized_text, keywords)
            if score > best_score:
                best_goal = goal
                best_score = score
                reasons = [f"target_goal=inferred:{goal}"]
        if best_score == 0.0:
            reasons = ["target_goal=inferred:awareness"]
        return best_goal, reasons

    def _infer_target_audience(
        self,
        normalized_text: str,
        explicit_audience: list[str],
    ) -> tuple[list[str], list[str]]:
        inferred = list(explicit_audience)
        reasons = [f"target_audience=explicit:{','.join(explicit_audience)}"] if explicit_audience else []
        for audience_tag, keywords in AUDIENCE_RULES.items():
            if self._keyword_score(normalized_text, keywords) > 0.0:
                inferred.append(audience_tag)
        if not inferred:
            inferred.append("general_public")
            reasons.append("target_audience=inferred:general_public")
        else:
            reasons.append(f"target_audience=inferred:{','.join(_dedupe_preserve_order(inferred))}")
        return _dedupe_preserve_order(inferred), reasons

    def _normalize_constraints(self, raw_constraints: dict[str, Any], normalized_text: str) -> EventConstraints:
        constraints = dict(raw_constraints)
        if "risk_level" not in constraints:
            constraints["risk_level"] = self._infer_risk_level(normalized_text)
        if "allowed_platforms" not in constraints:
            constraints["allowed_platforms"] = ["weibo_simulated"]
        return EventConstraints.model_validate(constraints)

    def _infer_risk_level(self, normalized_text: str) -> str:
        if any(keyword in normalized_text for keyword in HIGH_RISK_KEYWORDS):
            return "high"
        if any(keyword in normalized_text for keyword in MEDIUM_RISK_KEYWORDS):
            return "medium"
        return "low"

    def _extract_keywords(self, normalized_text: str, explicit_audience: list[str]) -> list[str]:
        keywords: list[str] = []
        for rule_set in (EVENT_TYPE_RULES, GOAL_RULES, AUDIENCE_RULES):
            for candidate_keywords in rule_set.values():
                keywords.extend(self._matched_keywords(normalized_text, candidate_keywords))
        keywords.extend(explicit_audience)
        return _dedupe_preserve_order(keywords)

    def _matched_keywords(self, normalized_text: str, keywords: tuple[str, ...]) -> list[str]:
        return [keyword for keyword in keywords if self._fuzzy_contains(normalized_text, keyword)]

    def _keyword_score(self, normalized_text: str, keywords: tuple[str, ...]) -> float:
        if not normalized_text:
            return 0.0
        matched = self._matched_keywords(normalized_text, keywords)
        if not matched:
            return 0.0
        return len(matched) / len(keywords)

    def _fuzzy_contains(self, normalized_text: str, keyword: str) -> bool:
        if keyword in normalized_text:
            return True
        tokens = normalized_text.split()
        if not tokens:
            return False
        return any(fuzz.partial_ratio(keyword, token) >= 90 for token in tokens)

    def _make_event_id(self) -> str:
        return f"event_{uuid4().hex[:10]}"
