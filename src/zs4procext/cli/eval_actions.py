import json
from typing import Any, Dict

import click

from zs4procext.evaluator import Evaluator


@click.command()
@click.argument("reference_dataset_path", type=str)
@click.argument("evaluated_dataset_path", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--similarity_threshold",
    default=0.8,
    help="Minimum threshold value to consider two actions as equals",
)
def eval_actions(
    reference_dataset_path: str,
    evaluated_dataset_path: str,
    output_file_path: str,
    similarity_threshold: float,
) -> None:
    evaluator = Evaluator(reference_dataset_path=reference_dataset_path)
    results: Dict[str, Any] = {
        "actions": evaluator.evaluate_actions(
            evaluated_dataset_path, threshold=similarity_threshold
        ),
        "sequence": evaluator.evaluate_actions_order(evaluated_dataset_path),
        "chemicals": evaluator.evaluate_chemicals(evaluated_dataset_path, threshold=similarity_threshold),
        "threshold": similarity_threshold,
    }
    results_json = json.dumps(results, indent=4)
    with open(output_file_path, "w") as f:
        f.write(results_json)


def main():
    eval_actions()


if __name__ == "__main__":
    main()
