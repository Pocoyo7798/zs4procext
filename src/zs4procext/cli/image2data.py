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
    vlm_model_name: Optional[str],
    vlm_model_parameters_path: Optional[str],
):
    start_time = time.time()
    print("Starting the image2data script...")
    print(f"Image folder: {image_folder}")
    print(f"Output file path: {output_file_path}")

    if prompt_structure_path is None:
        try:
            name = vlm_model_name.split("/")[-1]
            prompt_structure_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            print("Error: No matching template for model name.")
            return

    print(f"Using prompt structure: {prompt_structure_path}")

    individual_results = []


    file_list = os.listdir(image_folder)
    print(f"Found {len(file_list)} files in the folder. Filtering image files...")

    for file_name in file_list:
        file_extension = file_name.split(".")[-1].lower()
        if file_extension in {"png", "jpeg", "tiff"}:
            full_path = os.path.join(image_folder, file_name)
            print(f"Processing file: {full_path}")

            try:
                print("Initializing ImageExtractor...")
                extractor = ImageExtractor(
                    prompt_structure_path=prompt_structure_path,
                    prompt_schema_path=prompt_schema_path,
                    vlm_model_name=vlm_model_name,
                    vlm_model_parameters_path=vlm_model_parameters_path,
                )
                print("ImageExtractor initialized.")

                print(f"Extracting data from image: {file_name}")
                image_data = extractor.extract_image_info(full_path)

                image_dict = {file_name: image_data}
                individual_results.append(image_dict)
                print(f"Successfully extracted data for {file_name}: {image_data}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print("Merging individual results into a single dictionary...")
    merged_data = {}
    for item in individual_results:
        merged_data.update(item)

    print(f"Writing merged data to output file: {output_file_path}")
    with open(output_file_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

    total_time = time.time() - start_time
    print(f"Data successfully written to {output_file_path}")
    print(f"Total execution time: {total_time / 60:.2f} minutes")


def main():
    image2data()


if __name__ == "__main__":
    main()