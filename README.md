# abc_reading Workspace

这个目录是后续针对 `abc_reading` 数据集继续工作的专用工作区。

## 目录结构

```text
abc_reading_workspace/
  README.md
  data/
    raw/
      abc_reading_profile.graph.anon
      abc_reading_interaction.graph.anon
      abc_reading_product_info.json
    derived/
      abc_reading_profile_with_neighbors.graph.anon
  scripts/
    enrich_neighbors.py
```

## 数据说明

- `raw/abc_reading_profile.graph.anon`
  - 原始用户画像数据
- `raw/abc_reading_interaction.graph.anon`
  - 原始互动数据
- `raw/abc_reading_product_info.json`
  - 从总 `product_info.jsonl` 中筛出的 `abc_reading` 元数据
- `derived/abc_reading_profile_with_neighbors.graph.anon`
  - 在原始画像基础上新增邻居关系属性后的文件

## 邻居增强脚本

重新生成带 `neighbors` 的画像文件：

```bash
python scripts/enrich_neighbors.py
```

默认输入：

- `data/raw/abc_reading_profile.graph.anon`
- `data/raw/abc_reading_interaction.graph.anon`

默认输出：

- `data/derived/abc_reading_profile_with_neighbors.graph.anon`

## 邻居字段说明

每个用户会新增：

- `graph_attributes`
  - `neighbor_count`
  - `engaged_by_neighbor_count`
  - `engaged_to_neighbor_count`
  - `mutual_neighbor_count`
  - `self_interaction_count`
  - `received_*` / `made_*` 统计
  - `isolated`
- `neighbors`
  - `neighbor_id`
  - `relation`
  - `received_comment_count`
  - `received_repost_count`
  - `made_comment_count`
  - `made_repost_count`
  - `total_interaction_count`

关系语义：

- `engaged_by`：对方在当前用户帖子下互动
- `engaged_to`：当前用户去对方帖子下互动
- `mutual`：双方都有过互动
