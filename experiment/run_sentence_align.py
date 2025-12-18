import os
import yaml
import subprocess
import sys


def run_experiment(name, overrides):
    # Load base config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides (nested update)
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = update(config, overrides)

    # Save temp config
    temp_path = f"configs/temp_{name}.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(config, f)

    # Ensure result dir exists
    os.makedirs(os.path.join("results", name), exist_ok=True)
    log_path = os.path.join("results", name, "console.log")

    print(f"Running {name} ... log -> {log_path}")
    with open(log_path, "w") as log_f:
        # Tee subprocess output to both console and file (line-buffered).
        proc = subprocess.Popen(
            [
                "python3",
                "-u",
                "-m",
                "src.training.train",
                "--config",
                temp_path,
                "--exp_name",
                name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
            raise
        finally:
            proc.wait()

    # Cleanup temp
    if os.path.exists(temp_path):
        os.remove(temp_path)


def main():
    """
    Four variants:
      1) sentence_pretrained=True,  align_text_source=fine
      2) sentence_pretrained=True,  align_text_source=coarse
      3) sentence_pretrained=False, align_text_source=fine
      4) sentence_pretrained=False, align_text_source=coarse

    Classification tokens仅用图像（fusion_include_image=True, 其余False），
    对齐则使用句向量。若使用预训练权重，请将 all-MiniLM-L6-v2 模型目录
    放在项目根目录下或相应路径，并在 config 中指定。
    """
    base_overrides = {
        "data": {
            "batch_size": 16,
            "num_workers": 2,
        },
        "model": {
            "method_option": "alignment",
            "use_sentence_encoder": True,
            "fusion_include_image": True,
            "fusion_include_coarse": False,
            "fusion_include_fine": False,
            # sentence alignment uses raw texts; disable TextVAE branches to reduce memory
            "use_coarse": False,
            "use_fine": False,
            # 将预训练模型目录放在项目根目录下，或修改为绝对路径
            "sentence_encoder_name": "all-MiniLM-L6-v2",
            "freeze_sentence_encoder": True,
            # Run sentence encoder on GPU for speed. If you hit CUDA OOM, set to "cpu".
            "sentence_encoder_device": "cuda",
            # Use a smaller vision backbone if CUDA memory is limited.
            "backbone": "vit_tiny_patch16_224",
        },
        "training": {
            "lambda_align": 0.1,
            "align_temperature": 0.07,
        },
    }

    experiments = [
        # 预训练句向量（默认路径 all-MiniLM-L6-v2）
        ("sentence_align_fine_pretrained", {"model": {"align_text_source": "fine", "sentence_pretrained": True}}),
        ("sentence_align_coarse_pretrained", {"model": {"align_text_source": "coarse", "sentence_pretrained": True}}),
        # 自训文本编码器（路径指向你训练好的 coarse/fine 目录）
        ("sentence_align_fine_scratch", {
            "model": {
                "align_text_source": "fine",
                "sentence_pretrained": False,
                "sentence_encoder_name": "train_sentence_encoder/output_fine",
            }
        }),
        ("sentence_align_coarse_scratch", {
            "model": {
                "align_text_source": "coarse",
                "sentence_pretrained": False,
                "sentence_encoder_name": "train_sentence_encoder/output_coarse",
            }
        }),
    ]

    for name, ov in experiments:
        merged_model = {**base_overrides["model"], **ov.get("model", {})}
        merged_train = {**base_overrides["training"], **ov.get("training", {})}
        overrides = {"model": merged_model, "training": merged_train}
        run_experiment(name, overrides)


if __name__ == "__main__":
    main()
