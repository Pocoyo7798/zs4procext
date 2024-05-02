import json
from typing import Any, Dict, Optional

from langchain_community.llms import VLLM
from pydantic import BaseModel


class ModelLLM(BaseModel):
    model_name: str
    model_parameters: Dict[str, Any] = {}
    model_library: str = "vllm"
    model: Optional[VLLM] = None

    def vllm_load_model(self) -> None:
        """Load a model using vllm library"""
        if self.model_parameters == {}:
            self.model = VLLM(model=self.model_name)
        else:
            self.model = VLLM(
                model=self.model_name,
                best_of=self.model_parameters["best_of"],
                cache=self.model_parameters["cache"],
                callback_manager=self.model_parameters["callback_manager"],
                callbacks=self.model_parameters["callbacks"],
                download_dir=self.model_parameters["download_dir"],
                dtype=self.model_parameters["dtype"],
                frequency_penalty=self.model_parameters["frequency_penalty"],
                ignore_eos=self.model_parameters["ignore_eos"],
                logprobs=self.model_parameters["logprobs"],
                max_new_tokens=self.model_parameters["max_new_tokens"],
                metadata=self.model_parameters["metadata"],
                n=self.model_parameters["n"],
                presence_penalty=self.model_parameters["presence_penalty"],
                stop=self.model_parameters["stop"],
                tags=self.model_parameters["tags"],
                temperature=self.model_parameters["temperature"],
                tensor_parallel_size=self.model_parameters["tensor_parallel_size"],
                top_k=self.model_parameters["top_k"],
                top_p=self.model_parameters["top_p"],
                trust_remote_code=self.model_parameters["trust_remote_code"],
                use_beam_search=self.model_parameters["use_beam_search"],
                vllm_kwargs={
                    "gpu_memory_utilization": self.model_parameters[
                        "gpu_memory_utilization"
                    ],
                    "seed": self.model_parameters["seed"],
                    "enforce_eager": self.model_parameters["enforce-eager"]
                },
            )

    def load_model_parameters(self, file_path: str) -> None:
        """Load the model paramaters from a json file

        Args:
            file_path: Path to the file containing the model paramaters
        """
        with open(file_path, "r") as f:
            self.model_parameters = json.load(f)

    def run_single_prompt(self, prompt) -> str:
        if self.model is None:
            raise AttributeError("The LLM model is not loaded")
        return self.model(prompt)
