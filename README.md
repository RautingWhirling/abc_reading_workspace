# abc_reading Workspace

This workspace is focused on the `abc_reading` dataset and the baseline pipeline for
influence event distribution strategy generation.

## Structure

```text
abc_reading_workspace/
  data/
    raw/
      abc_reading_profile.graph.anon
      abc_reading_interaction.graph.anon
      abc_reading_product_info.json
    derived/
      abc_reading_profile_with_neighbors.graph.anon
  src/
    influence_strategy/
      data_loader.py
      event_parser.py
      feature_builder.py
      scorer.py
      selector.py
      models.py
  tests/
  scripts/
    enrich_neighbors.py
  environment.yml
  pyproject.toml
```

## Environment

This project uses `conda` for environment management.

```powershell
conda env create -f environment.yml
conda activate abc-reading-strategy
```

## Implemented Modules

### `data_loader`

Loads and validates:

- product context
- raw user profiles
- raw interaction records
- neighbor-enriched profiles

Main class:

```python
from influence_strategy.data_loader import DataLoader

loader = DataLoader(".")
summary = loader.build_summary()
bundle = loader.load_dataset_bundle()
```

### `event_parser`

Parses natural language or dictionary input into a structured event object.

Main class:

```python
from influence_strategy.event_parser import RuleBasedEventParser

parser = RuleBasedEventParser()
event = parser.parse("希望围绕亲子阅读和英语启蒙做一次传播活动")
```

### `feature_builder`

Builds the first-pass node feature set for one parsed event.

Main class:

```python
from influence_strategy.data_loader import DataLoader
from influence_strategy.event_parser import RuleBasedEventParser
from influence_strategy.feature_builder import FeatureBuilder

loader = DataLoader(".")
event = RuleBasedEventParser().parse("希望围绕亲子阅读和英语启蒙做一次传播活动")
builder = FeatureBuilder()
result = builder.build_features(
    product_context=loader.load_product_context(),
    profiles=loader.load_profiles(),
    event=event,
    enriched_profiles=loader.load_enriched_profiles(),
    source_user_ids=set(loader.load_interactions(limit_records_per_source=0).keys()),
)
frame = builder.to_frame(result)
```

### `scorer`

Scores nodes with risk-aware ranking based on the built features.

Main class:

```python
from influence_strategy.scorer import Scorer

score_result = Scorer().score(result)
score_frame = Scorer().to_frame(score_result)
```

### `selector`

Selects the final candidate nodes and builds a role-aware shortlist plus fallbacks.

Main class:

```python
from influence_strategy.selector import Selector

selection_result = Selector().select(score_result)
selected_frame = Selector().to_frame(selection_result)
fallback_frame = Selector().to_frame(selection_result, bucket="fallback")
```

## Tests

```powershell
python -m pytest
```

## Simple Visualization

```powershell
python tests/visualize_feature_builder.py
python tests/visualize_scorer.py
python tests/visualize_selector.py
```

Preview files will be saved under `outputs/feature_builder/`, `outputs/scorer/`, and `outputs/selector/`.
