"""PHM unified HI pipeline CLI."""

import argparse
from pathlib import Path

from config import (
    BATCH_SIZE,
    DATA2_DIR,
    EPOCHS,
    LEARNING_RATE,
    STRIDE,
    TEST_DIR,
    TRAIN_DIR,
    WINDOW_SIZE,
    clear_feature_cache,
)
from evaluate import evaluate_test
from train import MODEL_PATH, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHM HI regression CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Train HI model (data + data2)")
    tp.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    tp.add_argument("--data2-dir", type=Path, default=DATA2_DIR)
    tp.add_argument("--model-path", type=Path, default=MODEL_PATH)
    tp.add_argument("--epochs", type=int, default=EPOCHS)
    tp.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    tp.add_argument("--lr", type=float, default=LEARNING_RATE)
    tp.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    tp.add_argument("--stride", type=int, default=STRIDE)
    tp.add_argument("--max-samples", type=int, default=None)
    tp.add_argument("--no-balanced", action="store_true")
    tp.add_argument("--no-cache", action="store_true")
    tp.add_argument("--full-cv", action="store_true",
                    help="Run leave-one-TDMS-case-out 4-fold ensemble (default: single-fold).")
    tp.add_argument("--val-case", type=str, default=None,
                    help="Hold-out TDMS case (single-fold). Omit to follow config.")
    tp.add_argument("--seed", type=int, default=None)
    tp.add_argument("--no-data2", action="store_true",
                    help="Exclude data2 cases from training (TDMS-only).")

    ep = sub.add_parser("evaluate", help="Inference on data/Test")
    ep.add_argument("--test-dir", type=Path, default=TEST_DIR)
    ep.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ep.add_argument("--output", type=Path, default=None)
    ep.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ep.add_argument("--no-cache", action="store_true")

    sub.add_parser("clear-cache", help="Clear per-case NPZ feature cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(
            original_dir=args.data_dir,
            data2_dir=args.data2_dir,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            window_size=args.window_size,
            stride=args.stride,
            max_samples=args.max_samples,
            balanced=not args.no_balanced,
            use_cache=not args.no_cache,
            full_cv=args.full_cv,
            val_case=args.val_case,
            seed=args.seed,
            include_data2=not args.no_data2,
        )
    elif args.command == "evaluate":
        evaluate_test(
            test_dir=args.test_dir,
            model_path=args.model_path,
            output_path=args.output,
            window_size=args.window_size,
            use_cache=not args.no_cache,
        )
    elif args.command == "clear-cache":
        clear_feature_cache()


if __name__ == "__main__":
    main()
