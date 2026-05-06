from __future__ import annotations

from pathlib import Path
from typing import Any

from .event_parser import RuleBasedEventParser
from .feature_builder import FeatureBuilder
from .scorer import Scorer
from .selector import Selector
from .strategy_generator import StrategyGenerator
from .models import StrategyResult


class StrategyPipeline:
    def __init__(self, product_name: str = "abc_reading") -> None:
        self.product_name = product_name
        self.event_parser = RuleBasedEventParser(default_product_name=product_name)
        self.feature_builder = FeatureBuilder(product_name=product_name)
        self.scorer = Scorer()
        self.selector = Selector()
        self.strategy_generator = StrategyGenerator()

    def run(
        self,
        workspace_root: str | Path,
        event_input: str | dict[str, Any],
        profile_limit: int | None = None,
    ) -> StrategyResult:
        event = self.event_parser.parse(event_input)
        feature_result = self.feature_builder.build_from_loader(
            workspace_root=workspace_root,
            event=event,
            profile_limit=profile_limit,
        )
        score_result = self.scorer.score(feature_result)
        selection_result = self.selector.select(score_result)
        return self.strategy_generator.generate(selection_result)
