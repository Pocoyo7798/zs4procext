import time
from typing import List, Optional

import click

from zs4procext.extractor import SamplesExtractorFromText
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
    "--prompt_schema_path",
    default=None,
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
def text2samples(
    text_file_path: str,
    output_file_path: str,
    prompt_structure_path: Optional[str],
    prompt_schema_path: Optional[str],
    llm_model_name: str,
    llm_model_parameters_path: Optional[str],
):
    start_time = time.time()
    if prompt_structure_path is None:
        try:
            prompt_structure_path = TEMPLATE_REGISTRY[llm_model_name]
        except KeyError:
            pass
    extractor: SamplesExtractorFromText = SamplesExtractorFromText(
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
            extractor.retrieve_samples_from_text(text, ["notes", "note"])
        )
        with open(output_file_path, "a") as f:
            f.write(str(results) + "\n")
        count = count + 1
    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    text2samples()


if __name__ == "__main__":
    main()