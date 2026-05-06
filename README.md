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

## Tests

```powershell
python -m pytest
```
