import os
import time
from typing import List, Optional

import click

from xlmexlab.extractor import TableExtractor, Table2Blocks
from xlmexlab.prompt import TEMPLATE_REGISTRY


@click.command()
@click.argument("image_folder", type=str)
@click.argument("output_file_path", type=str)
@click.option(
    "--table_type",
    default="materials_characterization",
    help="Type of actions to considered. Options: All or pistachio or materials.",
)
@click.option(
    "--prompt_template_path",
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
    help="Name of the LLM used to process the tables",
)
@click.option(
    "--llm_model_parameters_path",
    default=None,
    help="Parameters of the LLM used to process the tables",
)
@click.option(
    "--table_schema_path",
    default=None,
    help="Parameters of the LLM used to process the tables",
)
def table2data(
    image_folder: str,
    output_file_path: str,
    table_type: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    llm_model_name: str,
    llm_model_parameters_path: Optional[str],
    table_schema_path: Optional[str],
):
    start_time = time.time()
    if prompt_template_path is None:
        try:
            name = llm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    extractor: TableExtractor = TableExtractor(
        table_type=table_type,
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        vlm_model_name=llm_model_name,
        vlm_model_parameters_path=llm_model_parameters_path,
    )
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    all_results = []

    file_list = sorted(os.listdir(image_folder))

    for file in file_list:
        extension = file.split(".")[-1].lower()

        if extension in {"png", "jpg", "jpeg", "tif", "tiff"}:
            file_path = os.path.join(image_folder, file)
            print(f"[INFO] Processing {file}")

            try:
                # extractor returns (image_file, list_of_lists)
                image_file, list_of_lists = extractor.extract_table_info(file_path)
                
                table = Table(
                    page=0,  # 0 = unknown
                    name=image_file,
                    block=list_of_lists
                )
                
                # Find headers and indexes
                table.find_collumn_headers()
                table.find_row_indexes()
                
                # Create result dict
                result = {
                    'image': image_file,
                    'page': table.page,
                    'name': table.name,
                    'block': table.block,
                    'type': table.type,
                    'collumn_headers': table.collumn_headers,
                    'row_indexes': table.row_indexes,
                    'number': table.number,
                    'legend': table.legend,
                    'box':table.box
                }
                
                all_results.append(result)
                print(f"[SUCCESS] Processed {file} - Found {len(list_of_lists)} rows")
                
            except Exception as e:
                print(f"[ERROR] Failed on {file}: {e}")
                all_results.append({
                    "image": file,
                    "tables": [],
                    "error": str(e),
                })
    
    # Save all results as JSON
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Processed {len(file_list)} files in {elapsed_time:.2f} seconds")
    print(f"[INFO] Results saved to: {output_file_path}")

def main():
    table2data()   


if __name__ == "__main__":
    main()
