import json
import os
import re
from sys import flags
from typing import Any, Dict, Iterator, List, Optional, Tuple

import click
import importlib_resources
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, PrivateAttr, validator

from xlmexlab import parser
from xlmexlab.llm import ModelLLM, ModelVLM
from xlmexlab.prompt import PromptFormatter
from xlmexlab.prompt_creation import PromptCreation, PromptCreationSchedule
from xlmexlab.parser_nanoparticles import ParserNanoparticle


class NanoparticlesExtractorParagraph(BaseModel):

    llm_model_name: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    prompt_schema_path: Optional[str] = None
    prompt_template_path: Optional[str] = None

    _prompt_creation: Optional[PromptCreation] = PrivateAttr(default=None)
    _prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)
    _nanoparticles_parser: Optional[ParserNanoparticle] = PrivateAttr(default=None)

    # FIX: declared as PrivateAttr so Pydantic manages them correctly
    _extracted_flags: Optional[dict] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        print("\n  [EXTRACTOR] model_post_init starting...")

        # --- LLM model parameters ---
        if self.llm_model_parameters_path is None:
            llm_param_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/model_parameters"
                / "llm_default_params.json"
            )
            print(f"  [EXTRACTOR] No llm_model_parameters_path given, using default: {llm_param_path}")
        else:
            llm_param_path = self.llm_model_parameters_path
            print(f"  [EXTRACTOR] Using llm_model_parameters_path: {llm_param_path}")

        # --- Load LLM ---
        model_name = self.llm_model_name or "Llama2-70B-chat-hf"
        print(f"  [EXTRACTOR] Loading ModelLLM with model_name='{model_name}'...")
        self._llm_model = ModelLLM(model_name=model_name)
        self._llm_model.load_model_parameters(llm_param_path)
        self._llm_model.vllm_load_model()
        print(f"  [EXTRACTOR] ModelLLM loaded successfully.")

        # --- Parser ---
        self._nanoparticles_parser = ParserNanoparticle()
        print(f"  [EXTRACTOR] ParserNanoparticle ready.")
        print(f"  [EXTRACTOR] model_post_init complete.\n")

    def extract_text_info(self, text: str):
        print(f"\n  [EXTRACTOR.extract_text_info] Called.")
        print(f"  [EXTRACTOR.extract_text_info] text preview: '{text[:120]}...'")
        print(f"  [EXTRACTOR.extract_text_info] _extracted_flags: {self._extracted_flags}")
        print(f"  [EXTRACTOR.extract_text_info] prompt_schema_path: {self.prompt_schema_path}")

        if self.prompt_schema_path is None:
            print(f"  [EXTRACTOR.extract_text_info] Building prompt via PromptCreation...")
            self._prompt_creation = PromptCreation()

            # FIX: use `text` argument, NOT self._paragraph (which was always None)
            print(f"  [EXTRACTOR.extract_text_info] Calling build_extraction_prompt_json with text and flags...")
            prompt_dict, targets = self._prompt_creation.build_extraction_prompt_json(
                text,                   # ← FIXED: was self._paragraph (always None)
                self._extracted_flags
            )
            print(f"  [EXTRACTOR.extract_text_info] prompt_dict keys: {list(prompt_dict.keys())}")
            print(f"  [EXTRACTOR.extract_text_info] targets: {targets}")
        else:
            # If you have a schema path, handle it here
            raise NotImplementedError(
                f"prompt_schema_path='{self.prompt_schema_path}' handling is not implemented. "
                f"Set prompt_schema_path=None to use the default PromptCreation flow."
            )

        print(f"\n  [EXTRACTOR.extract_text_info] Building PromptFormatter...")
        self._prompt = PromptFormatter(**prompt_dict)
        self._prompt.model_post_init(self.prompt_template_path)
        self._nanoparticles_parser._parameters = targets

        print(f"\n  [EXTRACTOR.extract_text_info] Formatting final prompt with text...")
        prompt = self._prompt.format_prompt(f"'{text}'")
        print(f"\n  [EXTRACTOR.extract_text_info]  PROMPT SENT TO LLM")
        print(prompt)
        print(f"  [EXTRACTOR.extract_text_info] \n")

        print(f"  [EXTRACTOR.extract_text_info] Running LLM inference...")
        data_response = self._llm_model.run_single_prompt(prompt).strip()
        print(f"\n  [EXTRACTOR.extract_text_info] LLM RAW RESPONSE ")
        print(data_response)
        print(f"  [EXTRACTOR.extract_text_info] \n")

        print(f"  [EXTRACTOR.extract_text_info] Running parser.replace()...")
        final_answer = self._nanoparticles_parser.replace(
            self._extracted_flags,
            data_response
        )
        print(f"  [EXTRACTOR.extract_text_info] Parser output: {final_answer}")

        return final_answer
    
    def extract_schedule_info(self, text: str, data_response: str):
        if (self._extracted_flags.get("dose_group") is True):
            x = self._nanoparticles_parser.parse_response(data_response)
            print(f"\n  [EXTRACTOR.extract_schedule_info] Parsed LLM response into dict: {x}")
            drug_names = []
            for item in x.get("dose_group", []):
                drug_name = item.get("drug_name")
                print(f"  [EXTRACTOR.extract_schedule_info] Extracted drug_name: {drug_name}")

                if drug_name is not None:
                    drug_names.append(drug_name)

            if drug_names:
                print(f"\n  [EXTRACTOR.extract_schedule_info] Called.")
                print(f"  [EXTRACTOR.extract_schedule_info] text preview: '{text[:120]}...'")
                print(f"  [EXTRACTOR.extract_schedule_info] _extracted_flags: {self._extracted_flags}")
                print(f"  [EXTRACTOR.extract_schedule_info] prompt_schema_path: {self.prompt_schema_path}")

                if self.prompt_schema_path is None:
                    print(f"  [EXTRACTOR.extract_schedule_info] Building prompt via PromptCreation...")
                    self._prompt_creation = PromptCreationSchedule()

                    # FIX: use `text` argument, NOT self._paragraph (which was always None)
                    print(f"  [EXTRACTOR.extract_schedule_info] Calling build_extraction_prompt_json with text and flags...")
                    prompt_dict_s  = self._prompt_creation.build_extraction_prompt_json(
                        text,                   # ← FIXED: was self._paragraph (always None)
                        drug_names
                    )
                    print(f"  [EXTRACTOR.extract_schedule_info] prompt_dict keys: {list(prompt_dict_s.keys())}")
                else:
                    # If you have a schema path, handle it here
                    raise NotImplementedError(
                        f"prompt_schema_path='{self.prompt_schema_path}' handling is not implemented. "
                        f"Set prompt_schema_path=None to use the default PromptCreation flow."
                    )

                print(f"\n  [EXTRACTOR.extract_schedule_info] Building PromptFormatter...")
                self._prompt = PromptFormatter(**prompt_dict_s)
                self._prompt.model_post_init(self.prompt_template_path)

                print(f"\n  [EXTRACTOR.extract_schedule_info] Formatting final prompt with text...")
                prompt = self._prompt.format_prompt(f"'{text}'")
                print(f"\n  [EXTRACTOR.extract_schedule_info]  PROMPT SENT TO LLM")
                print(prompt)
                print(f"  [EXTRACTOR.extract_schedule_info] \n")

                print(f"  [EXTRACTOR.extract_schedule_info] Running LLM inference...")
                data_response_s = self._llm_model.run_single_prompt(prompt).strip()
                print(f"\n  [EXTRACTOR.extract_schedule_info] LLM RAW RESPONSE ")
                print(data_response_s)
                print(f"  [EXTRACTOR.extract_schedule_info] \n")

                return data_response_s