"""
Benchmark SOTA KNN defense for NSL_KDD.

Flow for each noisy input CSV:
1) KNN label sanitization (k=15, eta=0.5 by default)
2) model_training.py --model all
3) Collect Accuracy/F1-Macro summary to one CSV under reports/sota
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

MODEL_ORDER = ["XGB", "CATB", "BAGGING", "HISTGBM", "GBM", "LGBM", "RF", "DNN"]
NOISE_FILE_PATTERN = re.compile(r"^nslkdd_train_clean_merged_noise_(\d+)$", flags=re.IGNORECASE)


def run_step(step_name: str, cmd: list[str], log_file: Path) -> str:
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
    rows: list[dict] = []
    pattern = re.compile(
        r"\b(XGB|CATB|BAGGING|HISTGBM|GBM|LGBM|RF|DNN)\s*-\s*"
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


def extract_noise_pct(path: Path) -> int | None:
    m = NOISE_FILE_PATTERN.match(path.stem)
    return int(m.group(1)) if m else None


def collect_noisy_files(input_dir: Path, noise_rates: set[int] | None) -> list[Path]:
    candidates = sorted(input_dir.glob("*.csv"))
    out: list[Path] = []
    for p in candidates:
        s = p.stem.lower()
        if not NOISE_FILE_PATTERN.match(s):
            continue
        if any(tag in s for tag in ["_dae_", "_recovered", "_aug_", "_knn_sanitized", "_remaining_after_recover"]):
            continue
        pct = extract_noise_pct(p)
        if pct is None:
            continue
        if noise_rates is not None and pct not in noise_rates:
            continue
        out.append(p)
    return out


def parse_noise_rates(text: str | None) -> set[int] | None:
    if text is None:
        return None
    vals: set[int] = set()
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if v <= 0 or v > 100:
            raise ValueError(f"Invalid noise percentage: {v}")
        vals.add(v)
    return vals if vals else None


def run_training(python_exe: str, training_script: Path, train_csv: Path, output_dir: Path, log_file: Path) -> list[dict]:
    out = run_step(
        f"model_training ({train_csv.stem})",
        [
            python_exe,
            str(training_script),
            "--model",
            "all",
            "-r",
            "nslkdd",
            "--train-in",
            str(train_csv),
            "--log-level",
            "INFO",
            "--output-dir",
            str(output_dir),
        ],
        log_file,
    )
    return parse_training_summary(out)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Benchmark SOTA KNN defense for NSL_KDD")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--input-dir", type=str, default=str(script_dir / "resources" / "NSLKDD" / "clean_merged"))
    parser.add_argument("--noise-rates", type=str, default="30,40,50,60,70", help="Optional filter, e.g. '30,40,50'")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--only-benign", action="store_true", help="Run KNN sanitization only on current Benign rows")
    parser.add_argument("--sota-dir", type=str, default=str(script_dir / "reports" / "sota"))
    parser.add_argument("--summary-csv", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    noise_rates = parse_noise_rates(args.noise_rates)
    noisy_files = collect_noisy_files(input_dir, noise_rates)
    if not noisy_files:
        raise SystemExit(f"No noisy files found in: {input_dir}")

    python_exe = args.python_exe
    sota_dir = Path(args.sota_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = sota_dir / timestamp
    root_dir.mkdir(parents=True, exist_ok=True)

    knn_script = script_dir / "knn_label_sanitization.py"
    training_script = script_dir / "training" / "model_training.py"
    if not knn_script.exists() or not training_script.exists():
        raise SystemExit("Required script not found (knn_label_sanitization.py/training/model_training.py)")

    all_rows: list[dict] = []

    for noisy_csv in noisy_files:
        noise_pct = extract_noise_pct(noisy_csv)
        tag = f"noise_{noise_pct}" if noise_pct is not None else noisy_csv.stem

        phase_dirs = {
            "01_sanitization": root_dir / tag / "01_knn_sanitization",
            "02_training": root_dir / tag / "02_model_training",
        }
        for phase_dir in phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)

        sanitized_csv = phase_dirs["01_sanitization"] / f"{tag}_knn_sanitized.csv"
        knn_report = phase_dirs["01_sanitization"] / f"{tag}_knn_report.json"

        knn_cmd = [
            python_exe,
            str(knn_script),
            "--input",
            str(noisy_csv),
            "--output",
            str(sanitized_csv),
            "--report",
            str(knn_report),
            "--resource",
            "nslkdd",
            "--k",
            str(args.k),
            "--eta",
            str(args.eta),
        ]
        if args.only_benign:
            knn_cmd.append("--only-benign")

        run_step(
            f"knn_sanitization ({tag})",
            knn_cmd,
            phase_dirs["01_sanitization"] / f"{tag}_knn_sanitization.log",
        )

        metrics = run_training(
            python_exe,
            training_script,
            sanitized_csv,
            phase_dirs["02_training"],
            phase_dirs["02_training"] / f"{tag}_model_training.log",
        )

        for row in metrics:
            row["noise_pct"] = noise_pct
            row["stage"] = "sota_knn"
            row["input_noisy_file"] = str(noisy_csv)
            row["sanitized_train_file"] = str(sanitized_csv)
            row["knn_report"] = str(knn_report)
            row["only_benign"] = bool(args.only_benign)
            all_rows.append(row)

    summary_csv = Path(args.summary_csv) if args.summary_csv else (root_dir / "sota_knn_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        all_rows,
        columns=[
            "noise_pct",
            "stage",
            "model",
            "accuracy",
            "f1_macro",
            "only_benign",
            "input_noisy_file",
            "sanitized_train_file",
            "knn_report",
        ],
    ).to_csv(summary_csv, index=False)

    print("=" * 100)
    print("[DONE] SOTA KNN BENCHMARK COMPLETED (NSL_KDD)")
    print(f"[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] SOTA folder: {root_dir}")


if __name__ == "__main__":
    main()
