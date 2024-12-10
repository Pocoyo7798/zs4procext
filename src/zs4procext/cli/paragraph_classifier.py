import time
from typing import List, Optional
import torch
import importlib_resources

import click

from zs4procext.classifier import ParagraphClassifier
from zs4procext.prompt import TEMPLATE_REGISTRY

@click.command()
@click.argument("text_file_path", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--prompt_structure_path",
    default=None,
    help="Path to the file containing the structure of the prompt",
)
@click.option(
    "--type",
    default="n2_physisorption",
    help="Path to the file containing the schema of the prompt",
)
@click.option(
    "--llm_model_name",
    default=None,
    help="Name of the LLM used to get the actions",
)
@click.option(
    "--llm_model_parameters_path",
    default=None,
    help="Parameters of the LLM used to get the actions",
)
def paragraph_classifier(
    text_file_path: str,
    output_file_path: str,
    prompt_structure_path: Optional[str],
    type: str,
    llm_model_name: str,
    llm_model_parameters_path: Optional[str],
):
    torch.cuda.empty_cache()
    start_time = time.time()
    if type == "n2_physisorption":
        prompt_schema_path = str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "classify_n2_physisorption_schema.json"
    )
    elif type == "ftir_pyridine":
        prompt_schema_path = str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "classify_ftiv_pyridine_schema.json"
    )
    elif type == "multi_sample":
        prompt_schema_path = str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "classify_multi_sample_schema.json"
    )
    if prompt_structure_path is None:
        try:
            name = llm_model_name.split("/")[-1]
            print(name)
            prompt_structure_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    extractor: ParagraphClassifier = ParagraphClassifier(
        prompt_structure_path=prompt_structure_path,
        prompt_schema_path=prompt_schema_path,
        llm_model_name=llm_model_name,
        llm_model_parameters_path=llm_model_parameters_path,
    )
    extractor.model_post_init(None)
    with open(text_file_path, "r") as f:
        text_lines: List[str] = f.readlines()
    size = len(text_lines)
    count = 1
    for text in text_lines:
        print(f"text processed: {count}/{size}")
        results: str = str(
            extractor.classify_paragraph(text)
        )
        with open(output_file_path, "a") as f:
            f.write(str(results) + "\n")
        count = count + 1
    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    paragraph_classifier()


if __name__ == "__main__":
    main()