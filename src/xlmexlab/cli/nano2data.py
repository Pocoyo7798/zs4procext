import json
import os
import time
import logging
from typing import Any, Dict, List, Optional
import click
from dataclasses import asdict, is_dataclass

from xlmexlab.extractor_nanoparticles import NanoparticlesExtractorParagraph
from xlmexlab.nanoparticle_paragraph import NanoparticleExtractor
from xlmexlab.prompt import TEMPLATE_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_blocks(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f).get("blocks", [])


def extract_flags(extractor, text):
    result = extractor.extract(text)
    if is_dataclass(result):
        return asdict(result)
    return dict(result)


def count_true_flags(flags):
    return sum(1 for v in flags.values() if v is True)


def llm_should_run(flags):
    return count_true_flags(flags) > 0


def extract_llm_values(llm_extractor, text, flags):
    try:
        llm_extractor._extracted_flags = flags
        return llm_extractor.extract_text_info(text), None
    except Exception as e:
        return None, str(e)


def merge(flags, llm):
    merged = dict(flags)
    if llm:
        merged.update(llm)
    return merged


def process_blocks_stream(
    blocks,
    regex_extractor,
    llm_extractor,
    min_text_length,
    skip_llm
):
    """
    Process blocks and return hierarchical structure:
    header_1, header_2, paragraph_1 (with result), paragraph_2 (with result), 
    header_3, header_4, paragraph_3 (with result), ...
    """
    results = []
    paragraph_id = 1
    error_count = 0
    header_count = 0

    for b in blocks:

        if b.get("type") == "section_header":
            header_count += 1
            header_obj = {
                f"header_{header_count}": {
                    "content": b.get("content", "").strip(),
                    "page": b.get("page")
                }
            }
            results.append(header_obj)
            continue

        if b.get("type") != "paragraph":
            continue

        text = b.get("content", "").strip()

        if len(text) < min_text_length:
            continue

        try:
            flags = extract_flags(regex_extractor, text)

            llm_values = None
            llm_error = None

            if llm_extractor and not skip_llm and llm_should_run(flags):
                llm_values, llm_error = extract_llm_values(
                    llm_extractor, text, flags
                )

            merged = merge(flags, llm_values)

            # Create paragraph with its result
            paragraph_obj = {
                f"paragraph_{paragraph_id}": {
                    "id": paragraph_id,
                    "page": b.get("page"),
                    "text": text,
                    "extraction": {
                        "merged": merged
                    }
                }
            }
            results.append(paragraph_obj)
            paragraph_id += 1

        except Exception as e:
            error_count += 1
            logger.error(e)

    return results, error_count


@click.command()
@click.argument("paragraph_json", type=click.Path(exists=True))
@click.argument("output_file_path", type=click.Path())
@click.option("--prompt_template_path", default=None)
@click.option("--prompt_schema_path", default=None)
@click.option("--llm_model_name", default=None)
@click.option("--llm_model_parameters_path", default=None)
@click.option("--skip_llm", is_flag=True, default=False)
@click.option("--min_text_length", type=int, default=200, show_default=True)

def nanoparticles2data(
    paragraph_json,
    output_file_path,
    prompt_template_path,
    prompt_schema_path,
    llm_model_name,
    llm_model_parameters_path,
    skip_llm,
    min_text_length,
):

    start = time.time()

    print("\nLOADING FILE")
    blocks = load_blocks(paragraph_json)

    
    # EXTRACTORS
    regex_extractor = NanoparticleExtractor()

    llm_extractor = None
    if not skip_llm:
        try:
            template = TEMPLATE_REGISTRY.get(
                llm_model_name.split("/")[-1] if llm_model_name else "default"
            )

            llm_extractor = NanoparticlesExtractorParagraph(
                prompt_template_path=template,
                prompt_schema_path=prompt_schema_path,
                llm_model_name=llm_model_name,
                llm_model_parameters_path=llm_model_parameters_path,
            )

        except Exception as e:
            logger.warning(f"LLM disabled: {e}")

    results, error_count = process_blocks_stream(
        blocks=blocks,
        regex_extractor=regex_extractor,
        llm_extractor=llm_extractor,
        min_text_length=min_text_length,
        skip_llm=skip_llm
    )

    output = {
        "source": {
            "file": os.path.abspath(paragraph_json),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_paragraphs": sum(1 for r in results if any("paragraph_" in k for k in r.keys())),
        },
        "result": results  # Hierarchical: header_1, header_2, paragraph_1, paragraph_2, ...
    }

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nDONE")
    print("Paragraphs:", output["source"]["total_paragraphs"])
    print("Errors:", error_count)
    print("Time (min):", (time.time() - start) / 60)


def main():
    nanoparticles2data()


if __name__ == "__main__":
    main()