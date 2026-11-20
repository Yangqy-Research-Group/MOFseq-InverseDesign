"""Predict CH4/N2 selectivity and CH4 mixture uptake from MOFseq strings."""
import argparse
from pathlib import Path

from model import MOFSeqPredictor, load_predictor
from utils import ROOT, PREDICTION_COLUMNS, TASK_FILES, TARGET_COLUMNS, read_inputs, selected_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/example_mofseq.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/predictions/predictions.csv")
    parser.add_argument("--task", choices=["both", *TASK_FILES], default="both")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--checkpoint", type=Path, help="Optional single retrained best_model.pt; target is read from the checkpoint.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--strict-vocabulary", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.resolve() == args.input.resolve():
            raise FileExistsError(f"Choose a new output file: {args.output}")
        frame = read_inputs(args.input)
        sequences = frame["MOFseq"].tolist()
        unknowns = [set() for _ in sequences]
        if args.checkpoint is not None:
            retrained = MOFSeqPredictor(args.checkpoint, device_name=args.device)
            matching = [task for task, target in TARGET_COLUMNS.items() if retrained.target_cols == [target]]
            if len(matching) != 1 or args.task not in {"both", matching[0]}:
                raise ValueError("The retrained checkpoint target does not match the requested task.")
            models = [(matching[0], retrained)]
        else:
            models = [(task, load_predictor(task, args.models_dir, args.device)) for task in selected_tasks(args.task)]
        for task, predictor in models:
            current = [set(predictor.unknown_tokens(seq)) for seq in sequences]
            if args.strict_vocabulary and any(current):
                raise ValueError("Unknown tokens found. Check the input against data/vocab.csv.")
            unknowns = [left | right for left, right in zip(unknowns, current)]
            frame[PREDICTION_COLUMNS[task]] = predictor.predict(
                sequences, batch_size=args.batch_size
            ).reshape(-1)
        frame["unknown_token_count"] = [len(tokens) for tokens in unknowns]
        frame["unknown_tokens"] = [" ".join(sorted(tokens)) for tokens in unknowns]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Wrote {len(frame)} predictions to {args.output}")
        if any(unknowns):
            print("Warning: unknown tokens were mapped to <unk>; interpret those predictions cautiously.")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
