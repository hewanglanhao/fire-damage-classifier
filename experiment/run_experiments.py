import os
import yaml
import subprocess
import argparse
import concurrent.futures


def run_experiment(name, config_overrides, log_to_file=False):
    if not log_to_file:
        print(f"Running Experiment: {name}")

    # Load base config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    # Helper to update nested dict
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    config = update(config, config_overrides)

    # Save temp config
    temp_config_path = f"configs/temp_{name}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    # We use subprocess to ensure clean state for each run
    stdout_target = None
    stderr_target = None
    
    if log_to_file:
        # Create output dir for logs
        log_dir = os.path.join("results", name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "console.log")
        print(f"Starting {name}... Logs: {log_path}")
        stdout_target = open(log_path, "w")
        stderr_target = subprocess.STDOUT

    try:
        subprocess.run(
            [
                "python3",
                "-m",
                "src.training.train",
                "--config",
                temp_config_path,
                "--exp_name",
                name,
            ],
            check=True,
            stdout=stdout_target,
            stderr=stderr_target,
        )
        if log_to_file:
            print(f"Finished {name}")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {name} failed with error: {e}")
    finally:
        # Cleanup
        if stdout_target:
            stdout_target.close()
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def main():
    experiments = [
        # 1. ViT Baseline
        {
            "name": "1_baseline",
            "config": {
                "model": {"type": "baseline", "backbone": "vit_base_patch16_224"}
            },
        },
        # 1. CNN Baseline (ResNet)
        {
            "name": "1_baseline_cnn_resnet",
            "config": {
                "model": {"type": "baseline_cnn", "backbone": "resnet101"}
            },
        },
        # 2. Method A: VAE (Not aligned)
        {"name": "2_method_a_vae", "config": {"model": {"method_option": "vae"}}},
        # 2. Method B: Alignment (Preferred Method)
        {
            "name": "2_method_b_alignment",
            "config": {"model": {"method_option": "alignment"}},
        },
        # 3. Ablation: Only Coarse (Alignment)
        {
            "name": "3_only_coarse",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "use_coarse": True,
                    "use_fine": False,
                    "use_image": True,
                }
            },
        },
        # 3. Ablation: Only Fine (Alignment)
        {
            "name": "3_only_fine",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "use_coarse": False,
                    "use_fine": True,
                    "use_image": True,
                }
            },
        },
        # 4. Encoder: CNN (Alignment)
        {
            "name": "4_encoder_cnn",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "coarse_encoder_type": "cnn",
                    "fine_encoder_type": "cnn",
                }
            },
        },
        # 4. Encoder: ViT (Alignment)
        {
            "name": "4_encoder_vit",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "coarse_encoder_type": "vit",
                    "fine_encoder_type": "vit",
                }
            },
        },
        # 5. Fusion: Only z_img (Alignment)
        {
            "name": "5_align_fusion_img_only",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "use_coarse": True,
                    "use_fine": False,
                    "fusion_include_coarse": False,
                    "fusion_include_fine": False,
                    "fusion_include_image": True,
                }
            },
        },
        # 5. Fusion: Image + Coarse (Alignment)
        {
            "name": "5_align_fusion_img_coarse",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "use_coarse": True,
                    "use_fine": False,
                    "fusion_include_coarse": True,
                    "fusion_include_fine": False,
                    "fusion_include_image": True,
                }
            },
        },
        # 6. Innovation: Gated Fusion (Alignment)
        {
            "name": "6_gated_fusion",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "use_gated_fusion": True,
                }
            },
        },
        # 6. Innovation: Gated Fusion (Alignment) - Coarse Only
        {
            "name": "6_gated_fusion_coarse",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "use_gated_fusion": True,
                    "use_coarse": True,
                    "use_fine": False,
                }
            },
        },
        # 6. Innovation: Gated Fusion (Alignment) - Fine Only
        {
            "name": "6_gated_fusion_fine",
            "config": {
                "model": {
                    "method_option": "alignment",
                    "use_gated_fusion": True,
                    "use_coarse": False,
                    "use_fine": True,
                }
            },
        },
    ]

    parser = argparse.ArgumentParser(description="Run specific experiments")
    parser.add_argument(
        "--run",
        nargs="+",
        help="Names of experiments to run (supports partial matching, e.g. '1_baseline' or just '1')",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available experiments"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="Number of parallel experiments to run"
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for exp in experiments:
            print(f" - {exp['name']}")
        return

    experiments_to_run = []
    if args.run:
        for target in args.run:
            found = False
            for exp in experiments:
                if target in exp["name"]:
                    if exp not in experiments_to_run:
                        experiments_to_run.append(exp)
                    found = True
            if not found:
                print(f"Warning: No experiment found matching '{target}'")
    else:
        experiments_to_run = experiments

    if not experiments_to_run:
        print("No experiments selected to run.")
        return

    print(f"Selected {len(experiments_to_run)} experiments to run.")
    
    if args.jobs > 1:
        print(f"Running with {args.jobs} parallel jobs.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = []
            for exp in experiments_to_run:
                futures.append(
                    executor.submit(run_experiment, exp["name"], exp["config"], log_to_file=True)
                )
            
            # Wait for all to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An experiment execution failed: {e}")
    else:
        for exp in experiments_to_run:
            run_experiment(exp["name"], exp["config"])


if __name__ == "__main__":
    main()
