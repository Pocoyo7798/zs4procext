import time
from typing import List, Optional
import torch
import os

import click

from zs4procext.extractor import ActionExtractorFromText
from zs4procext.prompt import TEMPLATE_REGISTRY


@click.command()
@click.argument("text_file_path", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--actions_type",
    default="All",
    help="Type of actions to considered. Options: All or pistachio or materials.",
)
@click.option(
    "--post_processing",
    default=True,
    help="True if you want to process the LLM output, False otherwise",
)
@click.option(
    "--prompt_template_path",
    default=None,
    help="Path to the file containing the structure of the action prompt",
)
@click.option(
    "--prompt_schema_path",
    default=None,
    help="Path to the file containing the schema of the action prompt",
)
@click.option(
    "--chemical_prompt_schema_path",
    default=None,
    help="Path to the file containing the schema of the chemical prompt",
)
@click.option(
    "--wash_chemical_prompt_schema_path",
    default=None,
    help="Path to the file containing the schema of the wash chemical prompt",
)
@click.option(
    "--add_chemical_prompt_schema_path",
    default=None,
    help="Path to the file containing the schema of the add chemical prompt",
)
@click.option(
    "--solution_chemical_prompt_schema_path",
    default=None,
    help="Path to the file containing the schema of the  newsolution chemical prompt",
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
@click.option(
    "--elementar_actions",
    default=False,
    help="True to transform all actions into combinations of elementar actions, False otherwise",
)
def text2actions(
    text_file_path: str,
    output_file_path: str,
    actions_type: str,
    post_processing: bool,
    prompt_template_path: Optional[str],
    chemical_prompt_schema_path: Optional[str],
    prompt_schema_path: Optional[str],
    wash_chemical_prompt_schema_path: Optional[str],
    add_chemical_prompt_schema_path: Optional[str],
    solution_chemical_prompt_schema_path: Optional[str],
    llm_model_name: str,
    llm_model_parameters_path: Optional[str],
    elementar_actions: bool
):
    torch.cuda.empty_cache()
    start_time = time.time()
    if prompt_template_path is None:
        try:
            name = llm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    extractor: ActionExtractorFromText = ActionExtractorFromText(
        actions_type=actions_type,
        post_processing=post_processing,
        action_prompt_template_path=prompt_template_path,
        chemical_prompt_template_path=prompt_template_path,
        action_prompt_schema_path=prompt_schema_path,
        chemical_prompt_schema_path=chemical_prompt_schema_path,
        wash_chemical_prompt_schema_path=wash_chemical_prompt_schema_path,
        add_chemical_prompt_schema_path=add_chemical_prompt_schema_path,
        solution_chemical_prompt_schema_path=solution_chemical_prompt_schema_path,
        llm_model_name=llm_model_name,
        llm_model_parameters_path=llm_model_parameters_path,
        elementar_actions=elementar_actions
    )
    #extractor.model_post_init(None)
    with open(text_file_path, "r") as f:
        text_lines: List[str] = f.readlines()
    size = len(text_lines)
    count = 1
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    for text in text_lines:
        print(f"text processed: {count}/{size}")
        results: str = str(
            extractor.retrieve_actions_from_text(text, ["notes", "note"])
        )
        with open(output_file_path, "a") as f:
            f.write(str(results) + "\n")
        count = count + 1
    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    text2actions()


if __name__ == "__main__":
    main()
