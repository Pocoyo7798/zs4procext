import time
from typing import List, Optional

import click

from zs4procext.extractor import SamplesExtractorFromText

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
