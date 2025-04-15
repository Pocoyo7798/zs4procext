import time
from typing import List, Optional
import os
import json

import click

from zs4procext.extractor import ImageExtractor
from zs4procext.prompt import TEMPLATE_REGISTRY

@click.command()
@click.argument("image_folder", type=str)
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
    "--vlm_model_name",
    default=None,
    help="Name of the LLM used to process the tables",
)
@click.option(
    "--vlm_model_parameters_path",
    default=None,
    help="Parameters of the LLM used to process the tables",
)
def image2data(
    image_folder: str,
    output_file_path: str,
    prompt_structure_path: Optional[str],
    prompt_schema_path: Optional[str],
    vlm_model_name: str,
    vlm_model_parameters_path: Optional[str],
):
    start_time = time.time()
    if prompt_structure_path is None:
        try:
            name = vlm_model_name.split("/")[-1]
            prompt_structure_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    print(prompt_structure_path)
    extractor = ImageExtractor(
        prompt_structure_path=prompt_structure_path,
        prompt_schema_path=prompt_schema_path,
        vlm_model_name=vlm_model_name,
        vlm_model_parameters_path=vlm_model_parameters_path,
    )
    file_list = os.listdir(image_folder)
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)

    extracted_data = []

    for file in file_list:
        extension = file.split(".")[-1]
        if extension in {"png", "jpeg", "tiff"}:
            output = extractor.extract_image_info(os.path.join(image_folder, file))
            
            # Nest data by image name
            extracted_data.append({
                "image_name": file,
                "extracted_data": output
            })

    final_output = {item["image_name"]: item["extracted_data"] for item in extracted_data}

    with open(output_file_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"{(time.time() - start_time) / 60} minutes")


def main():
    image2data()

if __name__ == "__main__":
    main()