"""Create the Task-1 project layout without overwriting existing files."""

import argparse
from pathlib import Path


def create_file_if_missing(path: Path, content: str) -> str:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(content)
    except FileExistsError:
        return "preserved"
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the PRODIGY_ML_01 Task-1 directory structure."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory to create (defaults to the directory containing this script).",
    )
    target_dir = parser.parse_args().target_dir.expanduser().resolve()
    dataset_dir = target_dir / "dataset"
    source_dir = target_dir / "src"

    try:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True, exist_ok=True)

        files = {
            source_dir / "train_model.py": (
                '"""Task-1 model training script."""\n'
                "# Add the model training workflow here.\n"
            ),
            target_dir / "requirements.txt": "numpy\npandas\nscikit-learn\n",
            target_dir / "README.md": (
                "# PRODIGY_ML_01 - Task 1\n\n"
                "Place `train.csv` and `test.csv` in `dataset/`, install dependencies "
                "with `python -m pip install -r requirements.txt`, then run "
                "`python src/train_model.py` to evaluate the model and create "
                "`dataset/submission.csv`.\n"
            ),
        }

        statuses = {
            path: create_file_if_missing(path, content)
            for path, content in files.items()
        }
    except OSError as error:
        raise SystemExit(f"Could not create project structure at {target_dir}: {error}")

    print(f"Project structure ready at: {target_dir}")
    for path, status in statuses.items():
        print(f"  {status}: {path}")


if __name__ == "__main__":
    main()
