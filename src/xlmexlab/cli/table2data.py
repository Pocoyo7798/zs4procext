import os
import json
import time
import click
from typing import Optional, List, Dict, Any
from xlmexlab.extractor import TableExtractor, List2Headers, Table2Blocks
from xlmexlab.prompt import TEMPLATE_REGISTRY


def parse_header_rows_from_response(response: str, convert_to_index: bool = True) -> List[int]:
    """
    Parse the VLM response to extract header row numbers and convert to indices.
    
    Args:
        response: VLM response containing row numbers
        convert_to_index: If True, converts row numbers (1-based) to indices (0-based)
    
    Examples (with convert_to_index=True):
    - "rows 1, 2, 3" → [0, 1, 2]
    - "The headers are in rows 1 and 2" → [0, 1]
    - '{"header_rows": [1, 2]}' → [0, 1]
    
    Examples (with convert_to_index=False):
    - "rows 1, 2, 3" → [1, 2, 3]
    """
    import re
    
    header_rows = []
    
    # Try JSON parsing first
    try:
        data = json.loads(response)
        if "header_rows" in data:
            header_rows = data["header_rows"]
        elif "headers" in data:
            header_rows = data["headers"]
        elif "rows" in data:
            header_rows = data["rows"]
    except:
        pass
    
    # If JSON parsing didn't work, extract all numbers from the response
    if not header_rows:
        numbers = re.findall(r'\b\d+\b', response)
        header_rows = [int(n) for n in numbers]
    
    # Convert row numbers (1-based) to indices (0-based)
    if convert_to_index and header_rows:
        header_rows = [row - 1 for row in header_rows]
        # Remove negative indices (in case VLM returned 0)
        header_rows = [idx for idx in header_rows if idx >= 0]
    
    return header_rows


@click.command()
@click.argument("image_folder", type=str)
@click.argument("output_file_path", type=str)
@click.option("--table_type", default="All", help="Type of table to process")
@click.option("--prompt_template_path", default=None, help="Path to prompt template for stage 1")
@click.option("--prompt_schema_path", default=None, help="Path to prompt schema for stage 1")
@click.option("--vlm_model_name", default=None, help="Name of VLM model")
@click.option("--vlm_model_parameters_path", default=None, help="Path to VLM model parameters")
@click.option("--enable_header_refinement", is_flag=True, default=False, 
              help="Enable stage 2: use VLM to refine header detection")
@click.option("--header_prompt_template_path", default=None, 
              help="Path to prompt template for header refinement (stage 2)")
@click.option("--header_prompt_schema_path", default=None,
              help="Path to prompt schema for header refinement (stage 2)")
def extract_tables_chain(
    image_folder: str,
    output_file_path: str,
    table_type: str,
    prompt_template_path: Optional[str],
    prompt_schema_path: Optional[str],
    vlm_model_name: Optional[str],
    vlm_model_parameters_path: Optional[str],
    enable_header_refinement: bool,
    header_prompt_template_path: Optional[str],
    header_prompt_schema_path: Optional[str],
):
    """
    Two-stage table extraction pipeline:
    
    Stage 1: Extract table data from images
    Stage 2 (optional): Refine header detection using VLM
    """
    start_time = time.time()
    
    print("STAGE 1: Extracting tables from images")
    if prompt_template_path is None and vlm_model_name:
        try:
            name = vlm_model_name.split("/")[-1]
            prompt_template_path = TEMPLATE_REGISTRY[name]
        except KeyError:
            pass
    
    extractor = TableExtractor(
        table_type=table_type,
        prompt_template_path=prompt_template_path,
        prompt_schema_path=prompt_schema_path,
        vlm_model_name=vlm_model_name,
        vlm_model_parameters_path=vlm_model_parameters_path,
    )
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    stage1_results = []
    file_list = sorted(os.listdir(image_folder))
    
    for file in file_list:
        extension = file.split(".")[-1].lower()
        
        if extension not in {"png", "jpg", "jpeg", "tif", "tiff"}:
            continue
            
        file_path = os.path.join(image_folder, file)
        print(f"[STAGE 1] Processing {file}")
        
        try:
            image_file, list_of_lists = extractor.extract_table_info(file_path)
            
            table = Table2Blocks(
                page=0,
                name=image_file,
                block=list_of_lists
            )
            
            # Initial header/index detection
            table.find_collumn_headers()
            table.find_row_indexes()
            
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
                'box': table.box
            }
            
            stage1_results.append(result)
            print(f"[SUCCESS] Extracted {len(list_of_lists)} rows, "
                  f"headers: {table.collumn_headers}")
            
        except Exception as e:
            print(f"[ERROR] Failed on {file}: {e}")
            stage1_results.append({
                "image": file,
                "error": str(e),
                "block": [],
                "collumn_headers": [],
                "row_indexes": []
            })
    
    # Save Stage 1 results
    stage1_output = output_file_path.replace('.json', '_stage1.json')
    with open(stage1_output, 'w', encoding='utf-8') as f:
        json.dump(stage1_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n[STAGE 1] Complete! Results saved to: {stage1_output}")
    
    # STAGE 2: Refine Headers (Optional) 
    if not enable_header_refinement:
        print("\n[INFO] Header refinement disabled. Skipping Stage 2.")
        final_results = stage1_results
    else:
        print("STAGE 2: Refining header detection with VLM")
        
        header_extractor = List2Headers(
            table_type=table_type,
            prompt_template_path=prompt_template_path,
            prompt_schema_path=prompt_schema_path,
            vlm_model_name=vlm_model_name,
            vlm_model_parameters_path=vlm_model_parameters_path
        )
        header_extractor.model_post_init()
        
        final_results = []
        
        for result in stage1_results:
            if 'error' in result:
                final_results.append(result)
                continue
            
            image_path = result['image']
            block = result['block']
            
            if not block:
                print(f"[STAGE 2] Skipping {os.path.basename(image_path)} - empty block")
                final_results.append(result)
                continue
            
            print(f"[STAGE 2] Refining headers for {os.path.basename(image_path)}")
            
            try:
                # Pass the block to the VLM for header detection
                _, vlm_response = header_extractor.extract_table_info(
                    image_path,
                    extracted_data=block
                )
                
                print(f"[VLM Response] {vlm_response}")
                
                # Parse the response to get header row numbers
                header_rows = parse_header_rows_from_response(vlm_response)
                
                # Update the result
                result['collumn_headers'] = header_rows
                result['vlm_header_response'] = vlm_response
                
                print(f"[SUCCESS] Updated headers to: {header_rows}")
                
            except Exception as e:
                print(f"[ERROR] Header refinement failed for {image_path}: {e}")
                result['header_refinement_error'] = str(e)
            
            final_results.append(result)
    
    # Save final results
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    elapsed_time = time.time() - start_time

    print(f"[COMPLETE] Processed {len(file_list)} images in {elapsed_time:.2f} seconds")
    print(f"[COMPLETE] Final results saved to: {output_file_path}")


def main():
    extract_tables_chain()


if __name__ == "__main__":
    main()