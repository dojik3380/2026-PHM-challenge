"""PHM STFT + CNN + LSTM RUL 파이프라인 CLI.

예시:
    python main.py train
    python main.py evaluate
"""

import argparse
from pathlib import Path

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    MODELS_DIR,
    PREDICTION_PATH,
    STRIDE,
    TEAM_NAME,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    VALIDATION_PREDICTION_PATH,
    WINDOW_SIZE,
    clear_stft_cache,
)
from evaluate import evaluate_model, predict_validation
from train import train_model


DEFAULT_VALIDATION_DIR = VALIDATION_DIR if VALIDATION_DIR.exists() else TEST_DIR


def _team_output_path(output_path: Path, team_name: str | None) -> Path:
    if team_name:
        return output_path.with_name(f"{team_name}_validation.xlsx")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PHM STFT + CNN + LSTM RUL prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the STFT CNN+LSTM model")
    train_parser.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    train_parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    train_parser.add_argument("--epochs", type=int, default=EPOCHS)
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    train_parser.add_argument("--stride", type=int, default=STRIDE)
    train_parser.add_argument("--max-samples", type=int, default=None)
    train_parser.add_argument("--validation-data-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    train_parser.add_argument("--validation-output-path", type=Path, default=VALIDATION_PREDICTION_PATH)
    train_parser.add_argument("--team-name", type=str, default=TEAM_NAME)
    train_parser.add_argument("--skip-validation-output", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model")
    eval_parser.add_argument("--data-dir", type=Path, default=TEST_DIR)
    eval_parser.add_argument("--model-path", type=Path, default=None,
                             help="모델 파일 전체 경로. --model-name과 함께 쓰면 --model-name 우선.")
    eval_parser.add_argument("--model-name", type=str, default=None,
                             help="모델 이름(확장자 제외). 예: RUL_Baseline, RUL_TDMSOnly")
    eval_parser.add_argument("--output-path", type=Path, default=PREDICTION_PATH)
    eval_parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    eval_parser.add_argument("--stride", type=int, default=STRIDE)
    eval_parser.add_argument("--max-samples", type=int, default=None)

    validation_parser = subparsers.add_parser("predict-validation", help="Create validation RUL score Excel file")
    validation_parser.add_argument("--data-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    validation_parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    validation_parser.add_argument("--output-path", type=Path, default=VALIDATION_PREDICTION_PATH)
    validation_parser.add_argument("--team-name", type=str, default=TEAM_NAME)
    validation_parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    validation_parser.add_argument("--max-samples", type=int, default=None)
    
    cache_parser = subparsers.add_parser("clear-cache", help="Clear STFT cache directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        model_path = train_model(
            data_dir=args.data_dir,
            model_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            window_size=args.window_size,
            stride=args.stride,
            max_samples=args.max_samples,
        )
        if not args.skip_validation_output:
            predict_validation(
                data_dir=args.validation_data_dir,
                model_path=model_path,
                output_path=_team_output_path(args.validation_output_path, args.team_name),
                window_size=args.window_size,
            )
    elif args.command == "evaluate":
        if args.model_name:
            resolved_model_path = MODELS_DIR / f"{args.model_name}.pt"
        elif args.model_path:
            resolved_model_path = args.model_path
        else:
            resolved_model_path = MODEL_PATH
        evaluate_model(
            data_dir=args.data_dir,
            model_path=resolved_model_path,
            output_path=args.output_path,
            window_size=args.window_size,
            stride=args.stride,
            max_samples=args.max_samples,
        )
    elif args.command == "predict-validation":
        output_path = _team_output_path(args.output_path, args.team_name)
        predict_validation(
            data_dir=args.data_dir,
            model_path=args.model_path,
            output_path=output_path,
            window_size=args.window_size,
            max_samples=args.max_samples,
        )
    elif args.command == "clear-cache":
        clear_stft_cache()


if __name__ == "__main__":
    main()
