"""
Auto ablation runner for EdgeIIoT data-poisoning pipeline.

Pipeline order:
1) symmetric_label_noise.py
2) model_training.py (all models)
3) dae_kmeans_knn_benign_filter.py
4) model_training.py (all models)
5) dnn_recover_grid.py
6) model_training.py (all models)

It also extracts Accuracy and F1 (Macro) per model from model_training logs,
then saves a consolidated CSV summary for ablation reporting.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


MODEL_ORDER = ["XGB", "CATB", "BAGGING", "LGBM", "RF", "DNN"]


def run_step(step_name: str, cmd: list[str], log_file: Path) -> str:
    """Run one pipeline command, capture output, and persist a step log."""
    print("=" * 100)
    print(f"[RUN] {step_name}")
    print("[CMD] " + " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stdout or "") + (proc.stderr or "")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(combined, encoding="utf-8")

    if proc.returncode != 0:
        print(combined)
        raise SystemExit(f"Step failed: {step_name}. See log: {log_file}")

    print(f"[OK] {step_name} -> log: {log_file}")
    return combined

def parse_training_summary(text: str) -> list[dict]:
    """Extract per-model Accuracy and F1 (Macro) rows from training output."""
    rows: list[dict] = []
    pattern = re.compile(
        r"\b(XGB|CATB|BAGGING|LGBM|RF|DNN)\s*-\s*"
        r"Accuracy:\s*([0-9]*\.?[0-9]+),\s*F1\s*\(Macro\):\s*([0-9]*\.?[0-9]+)",
        flags=re.IGNORECASE,
    )
    found: dict[str, dict] = {}
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        model = m.group(1).upper()
        found[model] = {
            "model": model,
            "accuracy": float(m.group(2)),
            "f1_macro": float(m.group(3)),
        }

    for model in MODEL_ORDER:
        if model in found:
            rows.append(found[model])

    return rows


def parse_noise_rates(text: str) -> list[int]:
    """Parse and validate a comma-separated list of noise percentages."""
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        v = int(token)
        if v <= 0 or v > 100:
            raise ValueError(f"Invalid noise percentage: {v}. Must be in [1, 100].")
        values.append(v)
    if not values:
        raise ValueError("--noise-rates is empty")
    return values


def run_training(
    python_exe: str,
    training_script: Path,
    train_csv: Path,
    output_dir: Path,
    stage_name: str,
) -> list[dict]:
    """Train all models for one stage and return the parsed metrics rows."""
    log_file = output_dir / f"{stage_name}_model_training.log"
    cmd = [
        python_exe,
        str(training_script),
        "--model",
        "all",
        "-r",
        "edgeiot",
        "--train-in",
        str(train_csv),
        "--log-level",
        "INFO",
        "--output-dir",
        str(output_dir),
    ]
    out = run_step(f"model_training ({stage_name})", cmd, log_file)
    metrics = parse_training_summary(out)

    if not metrics:
        print(f"[WARN] Could not parse model summary for stage '{stage_name}'. Check log: {log_file}")
    else:
        print(f"[SUMMARY] {stage_name}")
        for row in metrics:
            print(
                f"  {row['model']:<12s} - Accuracy: {row['accuracy']:.4f}, "
                f"F1 (Macro): {row['f1_macro']:.4f}"
            )

    return metrics


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_clean_input = script_dir / "resources" / "edgeiot" / "clean_merged" / "edgeiot_train_clean_merged.csv"

    parser = argparse.ArgumentParser(description="Auto ablation pipeline runner for EdgeIIoT")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--noise-rates", type=str, default="30, 40, 50, 60, 70",
                        help="Noise percentages for symmetric_label_noise, e.g. '10,20,30,40,50,60'")
    parser.add_argument("--kmeans-budget", type=int, default=15000)
    parser.add_argument("--k-neighbors", type=int, default=15)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=str(script_dir / "reports" / "ablation_logs"))
    parser.add_argument("--clean-input", type=str, default=str(default_clean_input), help="Path to the clean EdgeIIoT training CSV")
    args = parser.parse_args()

    python_exe = args.python_exe
    noise_rates = parse_noise_rates(args.noise_rates)
    clean_input = Path(args.clean_input).expanduser()

    if not clean_input.exists():
        raise SystemExit(f"Clean input file not found: {clean_input}")

    # Create a fresh timestamped root folder for this run.
    base_log_dir = Path(args.log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = base_log_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INIT] Ablation pipeline started at {timestamp}")
    print(f"[INIT] Base log directory: {base_log_dir}")
    print(f"[INIT] Timestamped log directory: {log_dir}")

    symmetric_script = script_dir / "symmetric_label_noise.py"
    dae_script = script_dir / "dae_kmeans_knn_benign_filter.py"
    recover_script = script_dir / "dnn_recover_grid.py"
    training_script = script_dir / "model_training.py"

    # Verify the core scripts are available before running the pipeline.
    for req in [symmetric_script, dae_script, recover_script, training_script]:
        if not req.exists():
            raise SystemExit(f"Required script not found: {req}")

    all_rows: list[dict] = []

    for noise_pct in noise_rates:
        ratio = noise_pct / 100.0
        run_tag = f"noise_{noise_pct}"

        # Create phase-specific output directories for this noise level.
        phase_dirs = {
            "01_symmetric": log_dir / run_tag / "01_symmetric_label_noise",
            "02_training_noise": log_dir / run_tag / "02_model_training_after_noise",
            "03_dae": log_dir / run_tag / "03_dae_kmeans_knn",
            "04_training_dae": log_dir / run_tag / "04_model_training_after_dae",
            "05_recover": log_dir / run_tag / "05_dnn_recover",
            "06_training_recover": log_dir / run_tag / "06_model_training_after_recover",
        }
        
        for phase_dir in phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate the poisoned training set.
        noise_output = phase_dirs["01_symmetric"] / f"{run_tag}_data.csv"
        run_step(
            f"symmetric_label_noise ({run_tag})",
            [
                python_exe,
                str(symmetric_script),
                "--input",
                str(clean_input),
                "--noise-rate",
                str(ratio),
                "--keep-tracking",
                "--log-level",
                "INFO",
                "--output",
                str(noise_output),
            ],
            phase_dirs["01_symmetric"] / f"{run_tag}_symmetric_label_noise.log",
        )

        if not noise_output.exists():
            raise SystemExit(f"Symmetric output file not found: {noise_output}")

        # Step 2: Train all models on the poisoned data.
        for row in run_training(python_exe, training_script, noise_output, phase_dirs["02_training_noise"], f"{run_tag}_01_after_noise"):
            row["noise_pct"] = noise_pct
            row["stage"] = "01_after_noise"
            row["train_file"] = str(noise_output)
            all_rows.append(row)

        # Step 3: Apply the DAE + KMeans + KNN defense.
        dae_clean = phase_dirs["03_dae"] / f"{run_tag}_clean.csv"
        dae_noise = phase_dirs["03_dae"] / f"{run_tag}_noise.csv"
        
        run_step(
            f"dae_kmeans_knn_benign_filter ({run_tag})",
            [
                python_exe,
                str(dae_script),
                "--input",
                str(noise_output),
                "--kmeans-budget",
                str(args.kmeans_budget),
                "--k-neighbors",
                str(args.k_neighbors),
                "--log-level",
                "INFO",
                "--output-dir",
                str(phase_dirs["03_dae"]),
                "--name",
                run_tag,
            ],
            phase_dirs["03_dae"] / f"{run_tag}_dae_kmeans_knn.log",
        )

        if not dae_clean.exists() or not dae_noise.exists():
            raise SystemExit(
                "DAE stage output not found. Expected: "
                f"clean={dae_clean}, noise={dae_noise}"
            )

        # Step 4: Retrain after the DAE filtering stage.
        for row in run_training(python_exe, training_script, dae_clean, phase_dirs["04_training_dae"], f"{run_tag}_02_after_dae"):
            row["noise_pct"] = noise_pct
            row["stage"] = "02_after_dae"
            row["train_file"] = str(dae_clean)
            all_rows.append(row)

        # Step 5: Recover clean samples with the DNN-based recovery stage.
        recovered_clean = phase_dirs["05_recover"] / f"{run_tag}_recovered_final.csv"
        recovered_noise = phase_dirs["05_recover"] / f"{run_tag}_remaining_after_recover.csv"
        
        run_step(
            f"dnn_recover_grid ({run_tag})",
            [
                python_exe,
                str(recover_script),
                "--clean-in",
                str(dae_clean),
                "--noise-in",
                str(dae_noise),
                "--output-dir",
                str(phase_dirs["05_recover"]),
                "--name",
                run_tag,
            ],
            phase_dirs["05_recover"] / f"{run_tag}_dnn_recover.log",
        )

        if not recovered_clean.exists() or not recovered_noise.exists():
            raise SystemExit(
                "DNN recover output not found. Expected: "
                f"clean={recovered_clean}, noise={recovered_noise}"
            )

        # Step 6: Retrain on the recovered clean dataset.
        for row in run_training(python_exe, training_script, recovered_clean, phase_dirs["06_training_recover"], f"{run_tag}_03_after_recover"):
            row["noise_pct"] = noise_pct
            row["stage"] = "03_after_recover"
            row["train_file"] = str(recovered_clean)
            all_rows.append(row)

    if args.summary_csv:
        summary_csv = Path(args.summary_csv)
    else:
        summary_csv = log_dir / "ablation_f1_macro_summary.csv"

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary = pd.DataFrame(
        all_rows,
        columns=["noise_pct", "stage", "model", "accuracy", "f1_macro", "train_file"],
    )
    df_summary.to_csv(summary_csv, index=False)

    print("=" * 100)
    print("[DONE] ABLATION PIPELINE COMPLETED")
    print(f"[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] Logs folder: {log_dir}")


if __name__ == "__main__":
    main()
