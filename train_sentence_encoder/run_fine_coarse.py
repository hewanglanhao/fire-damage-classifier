import json
from pathlib import Path
from typing import Dict

from train_sentence_encoder.train import TrainingConfig, run_training


def main() -> None:
    """
    顺序训练基于 fine 文本和 coarse 文本的两个句子编码器，
    自动在各自的测试集上评估并保存结果。
    """
    base_output = Path("train_sentence_encoder")

    experiments = {
        "fine": base_output / "output_fine",
        "coarse": base_output / "output_coarse",
    }

    summary: Dict[str, Dict] = {}

    for text_field, out_dir in experiments.items():
        print("=" * 80)
        print(f"Training sentence encoder with text_field='{text_field}'")
        print(f"Output directory: {out_dir}")

        cfg = TrainingConfig(
            text_field=text_field,
            output_dir=str(out_dir),
        )
        metrics = run_training(cfg)
        summary[text_field] = metrics

    # 保存 fine / coarse 的汇总结果
    summary_path = base_output / "summary_fine_coarse.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"All experiments finished. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

