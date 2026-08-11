"""Report precision, recall, and F1 for the document extraction gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from semantica.explorer.document_evaluation import evaluate_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gold_file",
        nargs="?",
        type=Path,
        default=REPOSITORY_ROOT / "tests" / "explorer" / "document_extraction_gold.json",
    )
    parser.add_argument("--minimum-entity-f1", type=float, default=0.90)
    parser.add_argument("--minimum-relation-f1", type=float, default=0.90)
    args = parser.parse_args()

    report = evaluate_file(args.gold_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["entities"]["f1"] < args.minimum_entity_f1:
        return 1
    if report["relations"]["f1"] < args.minimum_relation_f1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
