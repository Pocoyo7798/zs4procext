import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import click
import importlib_resources
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, PrivateAttr, validator

from xlmexlab.llm import ModelLLM, ModelVLM

from xlmexlab.prompt import PromptFormatter

from xlmexlab.prompt_creation import PromptCreation
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

    _paragraph: Optional[str] = PrivateAttr(default=None)
    _extracted_flags: Optional[dict[str, Any]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.llm_model_parameters_path is None:
            llm_param_path = str(
                importlib_resources.files("xlmexlab")
                / "resources/model_parameters"
                / "llm_default_params.json"
            )
        else:
            llm_param_path = self.llm_model_parameters_path
        if self.prompt_schema_path is None:
            self._prompt_creation = PromptCreation()
            prompt_dict, targets = self._prompt_creation.build_extraction_prompt_json(self._paragraph, self._extracted_flags)
        self._prompt = PromptFormatter(**prompt_dict)
        self._prompt.model_post_init(self.prompt_template_path)
        if self.llm_model_name is None:
            self._llm_model = ModelLLM(model_name="Llama2-70B-chat-hf")
        else:
            self._llm_model = ModelLLM(model_name=self.llm_model_name)
        self._llm_model.load_model_parameters(llm_param_path)
        self._llm_model.vllm_load_model()
        self._nanopartciles_parser = ParserNanoparticle()
        self._nanoparticles_parser._parameters = targets

    def extract_text_info(self, text: str):

        prompt = self._prompt.format_prompt(f"'{text}'")
        data_response = self._llm_model.run_single_prompt(prompt).strip()

        final_answer = self._nanoparticles_parser.replace(
            self._extracted_flags,
            data_response
        )

        return final_answer