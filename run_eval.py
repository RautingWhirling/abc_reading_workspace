from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from influence_strategy.eval_hot_events import run_hot_event_evaluation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 eval 热点事件数据，生成包含五个维度和数字人选择说明的分发策略输出。",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT,
        help="项目工作目录，默认是当前项目根目录。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "eval" / "hot_event_opinion_variants.json",
        help="热点事件 eval JSON 文件，可以是单条对象或事件数组。",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        help="指定要运行的热点事件 ID；不指定时默认使用输入文件第一条。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval" / "output",
        help="输出目录，默认写入 eval/output。",
    )
    parser.add_argument(
        "--profile-limit",
        type=int,
        help="仅加载前 N 个 profile，便于快速调试。",
    )
    parser.add_argument(
        "--max-selected-nodes",
        type=int,
        default=5,
        help="主选数字人数量上限，默认 5。",
    )
    parser.add_argument(
        "--risk-level",
        choices=["low", "medium", "high"],
        help="覆盖事件风险等级；默认按热点领域做简单推断。",
    )
    parser.add_argument(
        "--campaign-window-hours",
        type=int,
        default=24,
        help="传播窗口时长，单位小时，默认 24。",
    )
    parser.add_argument(
        "--max-frequency-per-day",
        type=int,
        default=3,
        help="单个数字人每日触发频次上限，默认 3。",
    )
    parser.add_argument(
        "--allowed-platforms",
        type=str,
        help="允许的平台列表，使用逗号分隔，例如 weibo_simulated,optional_simulated。",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="关闭大模型内容生成，使用规则模板兜底。",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    allowed_platforms = None
    if args.allowed_platforms:
        allowed_platforms = [
            item.strip()
            for item in args.allowed_platforms.split(",")
            if item.strip()
        ]

    output_path, payload = run_hot_event_evaluation(
        workspace_root=args.workspace_root.resolve(),
        input_path=args.input.resolve(),
        output_dir=args.output_dir.resolve(),
        event_id=args.event_id,
        profile_limit=args.profile_limit,
        max_selected_nodes=args.max_selected_nodes,
        risk_level=args.risk_level,
        campaign_window_hours=args.campaign_window_hours,
        max_frequency_per_day=args.max_frequency_per_day,
        allowed_platforms=allowed_platforms,
        use_llm=not args.disable_llm,
    )

    console_summary = {
        "output_path": str(output_path),
        "event_name": payload["事件名称"],
        "selected_digital_human_ids": payload["选取数字人id组"],
        "selected_digital_human_count": len(payload["选取数字人id组"]),
    }
    print(json.dumps(console_summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
