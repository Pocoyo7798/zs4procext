import time
from typing import List, Optional
import os

import click

from zs4procext.extractor import TableExtractor
from zs4procext.prompt import TEMPLATE_REGISTRY

@click.command()
@click.argument("image_folder", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--table_type",
    default="materials_characterization",
    help="Type of actions to considered. Options: All or pistachio or materials.",
)
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
    "--vlm_model_name",
    default=None,
    help="Name of the LLM used to process the tables",
)
@click.option(
    "--vlm_model_parameters_path",
    default=None,
    help="Parameters of the LLM used to process the tables",
)
def table2data(
    image_folder: str,
    output_file_path: str,
    table_type: str,
    prompt_structure_path: Optional[str],
    prompt_schema_path: Optional[str],
    vlm_model_name: str,
    vlm_model_parameters_path: Optional[str],
):
    start_time = time.time()
    if prompt_structure_path is None:
        try:
            name = "/".split(vlm_model_name)[-1]
            prompt_structure_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    extractor: TableExtractor = TableExtractor(table_type=table_type, prompt_structure_path=prompt_structure_path, prompt_schema_path=prompt_schema_path, vlm_model_name=vlm_model_name, vlm_model_parameters_path=vlm_model_parameters_path)
    file_list = os.listdir(image_folder)
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    for file in file_list:
        extension = ".".split(file)
        if extension in {"png", "jpeg", "tiff"}:
            file_path = f"{image_folder}/{file}"
            extractor.extract_table_info(file_path)
    print(f"{(time.time() - start_time) / 60} minutes")
def main():
    table2data()


if __name__ == "__main__":
    main()