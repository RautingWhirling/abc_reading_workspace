# abc_reading 影响力事件分发策略原型

这是一个面向 `abc_reading` 数据集的轻量级原型项目，目标是把“输入一个事件，输出一份结构化传播策略”这条流水线先稳定做出来，后续再逐步接入大模型 API 做增强。

## 项目目标

- 输入事件描述
- 解析事件为结构化字段
- 基于 `abc_reading` 的用户画像、互动关系和邻居信息构建节点特征
- 对候选节点做风险感知评分
- 选择主选节点与备选节点
- 生成可直接落地到后续 API 的结构化传播策略

## 总体流程

```text
事件输入
  -> event_parser
  -> feature_builder
  -> scorer
  -> selector
  -> strategy_generator
  -> 结构化 JSON 策略
```

## 环境

当前使用 `conda` 管理环境。

```powershell
conda env create -f environment.yml
conda activate abc-reading-strategy
```

## 目录

```text
abc_reading_workspace/
  data/
    raw/
    derived/
  outputs/
  scripts/
  src/influence_strategy/
  tests/
  main.py
  environment.yml
  pyproject.toml
```

## 模块说明

| 模块 | 职责 | 主要输出 |
| --- | --- | --- |
| `data_loader` | 读取 `profile`、`interaction`、`neighbor_enriched_profile` | `DatasetBundle` |
| `event_parser` | 把自然语言事件整理成结构化事件 | `ParsedEvent` |
| `feature_builder` | 为每个节点构建可解释特征 | `FeatureBuildResult` |
| `scorer` | 计算风险感知分数 | `ScoreResult` |
| `selector` | 选择主选节点与备选节点 | `SelectionResult` |
| `strategy_generator` | 生成最终结构化传播策略 | `StrategyResult` |
| `pipeline` | 串起整条流水线 | `StrategyResult` |
| `main.py` | 命令行入口 | JSON 文件 + 控制台输出 |

## 数据使用范围

当前只使用 `abc_reading` 相关内容。

- `data/raw/abc_reading_profile.graph.anon`
- `data/raw/abc_reading_interaction.graph.anon`
- `data/raw/abc_reading_product_info.json`
- `data/derived/abc_reading_profile_with_neighbors.graph.anon`

## 运行方式

先跑测试：

```powershell
conda run -n abc-reading-strategy python -m pytest
```

再运行主流程：

```powershell
python main.py --event-text "希望围绕亲子阅读与英语启蒙做一次传播活动，提升讨论度并控制集中刷屏风险"
```

也可以用事件文件：

```powershell
python main.py --event-file samples\event.json
```

运行 `eval` 热点事件样例，并在 `eval/output` 下生成包含五个维度和数字人选择说明的结果：

```powershell
python run_eval.py
```

不指定 `--event-id` 时默认读取 `eval/hot_event_opinion_variants.json` 的前 10 条热点数据，并分别写入 `eval/output/<event_id>_strategy_output.json`；每条热点数据中的 `opinion_variants` 会作为 10 条不同叙述一起写入事件描述。只运行单条事件时可以指定：

```powershell
python run_eval.py --event-id hot_event_001
```

Windows 备注：

- 如果直接使用 `conda run -n abc-reading-strategy python main.py --event-text "中文事件"`，`conda` 本身可能因为终端编码报错
- 更稳妥的方式是先 `conda activate abc-reading-strategy` 再运行 `python main.py`
- 或者把事件写入 JSON 文件后使用 `--event-file`

可选参数：

- `--workspace-root`：工作目录
- `--risk-level`：覆盖风险等级
- `--max-selected-nodes`：覆盖主选节点数量
- `--max-frequency-per-day`：覆盖单节点日频次
- `--campaign-window-hours`：覆盖传播窗口
- `--allowed-platforms`：覆盖平台列表
- `--profile-limit`：仅加载前 N 个 profile 便于调试
- `--event-limit`：未指定 `--event-id` 时运行前 N 条热点事件，默认 10
- `--output`：指定输出 JSON 路径

## 输出结果

默认会写入：

```text
outputs/strategy/strategy_<event_id>.json
```

输出 JSON 的核心字段包括：

- `event`
- `product_context`
- `selection_summary`
- `summary`
- `stage_plans`
- `selected_nodes`
- `fallback_nodes`
- `strategy`

其中 `strategy` 里包含：

- `target_object`
- `objective`
- `time_plan`
- `frequency_plan`
- `platform_plan`
- `content_plan`
- `risk_control`
- `explainability`

`run_eval.py` 的输出默认写入：

```text
eval/output/<event_id>_strategy_output.json
```

该文件会额外整理：

- `five_dimensions`：分发对象、时间安排、频率安排、平台安排、内容安排
- `selected_digital_humans`：被选中分发的数字人信息、角色、阶段、评分、风险和选择理由
- `fallback_digital_humans`：备选数字人信息

说明：

- 控制台输出使用 ASCII-safe JSON，避免 Windows 终端编码问题
- 文件输出使用 UTF-8，可直接查看中文内容

## 可视化与调试

```powershell
python tests\visualize_feature_builder.py
python tests\visualize_scorer.py
python tests\visualize_selector.py
python tests\visualize_strategy_generator.py
```

## 当前规则逻辑

- `event_parser` 负责把事件整理成统一结构
- `feature_builder` 主要看节点影响力、扩散能力、话题匹配度、稳定性
- `scorer` 会额外引入风险惩罚
- `selector` 会优先保留不同角色的节点，保证首发、互动和扩散都有人选
- `strategy_generator` 会把节点结果翻译成阶段计划、频率计划、平台计划和内容计划

## 后续接入 LLM API 的改进点

建议后续把 LLM 放在“增强层”，不要一开始替代整条规则流水线。

- 事件解析增强：把自然语言事件解析成更完整的结构化字段
- 内容生成增强：为不同角色生成更自然的内容模板
- 策略解释增强：把规则型解释改成更自然的说明文本
- 风险审查增强：辅助识别敏感事件、过密触发和内容冲突
- API 服务化：用 `fastapi` 封装成可调用接口

建议后续再补的依赖：

- `openai`
- `python-dotenv`
- `httpx`
- `fastapi`
- `uvicorn`

## 备注

- 当前版本是可解释 baseline，不是完整生产系统
- 当前不做真实平台联动
- 当前不做复杂多智能体编排
- 当前重点是把“事件 -> 节点 -> 策略”这条链路先跑稳
