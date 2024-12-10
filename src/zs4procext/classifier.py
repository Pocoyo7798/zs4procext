import json
from typing import Any, Dict, Iterator, List, Optional
import os
import re

import importlib_resources
from pydantic import BaseModel, PrivateAttr
from zs4procext.llm import ModelLLM
from zs4procext.prompt import PromptFormatter

class ParagraphClassifier(BaseModel):
    llm_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    prompt_structure_path: Optional[str] = None
    prompt_schema_path: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    _prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        with open(self.prompt_schema_path, "r") as f:
            prompt_dict = json.load(f)
        self._prompt = PromptFormatter(**prompt_dict)
        self._prompt.model_post_init(self.prompt_structure_path)
        if self.llm_model_parameters_path is None:
            llm_param_path = str(
                importlib_resources.files("zs4procext")
                / "resources"
                / "vllm_default_params.json"
            )
        else:
            llm_param_path = self.llm_model_parameters_path
        self._llm_model = ModelLLM(model_name=self.llm_model_name)
        self._llm_model.load_model_parameters(llm_param_path)
        self._llm_model.vllm_load_model()

    def classify_paragraph(self, text) -> bool:
        prompt: str = self._prompt.format_prompt(text)
        print(prompt)
        response: str = self._llm_model.run_single_prompt(prompt).strip()
        print(response)
        answer_amount: List[str] = re.findall(r"\b(yes|Yes|no|No)\b", response)
        if len(answer_amount) == 0:
            result: bool = True
        elif answer_amount[0].lower() == "yes":
            result = True
        else:
            result = False
        return result