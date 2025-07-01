import json
from typing import Any, Dict, Optional

from langchain_community.llms import VLLM
from pydantic import BaseModel
from PIL import Image
import numpy as np
from vllm.lora.request import LoRARequest

import os

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
                    "enforce_eager": self.model_parameters["enforce-eager"],
                    "quantization": self.model_parameters["quantization"],
                    "max_model_len": self.model_parameters["max_model_len"]
                },
            )

    def load_model_parameters(self, file_path: str) -> None:
        """Load the model paramaters from a json file

        Args:
            file_path: Path to the file containing the model paramaters
        """
        with open(file_path, "r") as f:
            self.model_parameters = json.load(f)

    def run_single_prompt(self, prompt: str) -> str:
        if self.model is None:
            raise AttributeError("The LLM model is not loaded")
        return self.model(prompt)

class ModelVLM(BaseModel):
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
                image_input_type="pixel_values",
                image_token_id=self.model_parameters["image_token_id"],
                image_input_shape=self.model_parameters["image_input_shape"],
                image_feature_size=self.model_parameters["image_feature_size"],
                vllm_kwargs={
                    "gpu_memory_utilization": self.model_parameters[
                        "gpu_memory_utilization"
                    ],
                    "seed": self.model_parameters["seed"],
                    "enforce_eager": self.model_parameters["enforce-eager"],
                    "quantization": self.model_parameters["quantization"],
                    "max_model_len": self.model_parameters["max_model_len"],
                },
            )

    def load_model_parameters(self, file_path: str) -> None:
        """Load the model paramaters from a json file

        Args:
            file_path: Path to the file containing the model paramaters
        """
        with open(file_path, "r") as f:
            self.model_parameters = json.load(f)

    def run_image_single_prompt(self, prompt: str, image_path: str) -> str:
        pil_image = Image.open(image_path)
        new_prompt: Dict[str, Any] = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_image},
            }
        ]
        outputs = self.model.generate(new_prompt)
        final_response = ""
        for o in outputs:
            final_response += o[1][0][0].text
            break
        return final_response

    def run_image_single_prompt_rescale(self, prompt: str, image_path: str, scale: float = 1.0) -> str:
        pil_image = Image.open(image_path)
        if scale < 1.0:
            new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
            pil_image = pil_image.resize(new_size, Image.BILINEAR)

        new_prompt: Dict[str, Any] = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": pil_image},
            }
        ]

        outputs = self.model.generate(prompts=new_prompt) 
        final_response = ""
        for o in outputs:
            final_response += o[1][0][0].text
            break
        return final_response

from zs4procext.prompt import LORA_MODEL_CONFIGS

class InferencebyHF(BaseModel):
    model_name:str
    lora_path: Optional[str] = None
    
    processor: Optional[Any]=None
    process_vision_info: Optional[Any]=None


    def model_post_init(self, __context: Any) -> None:
        model_name_key = os.path.basename(self.model_name.rstrip("/"))

        config = LORA_MODEL_CONFIGS[model_name_key]

        model_path = config["model_class"]
        module_name, class_name = model_path.rsplit(".", 1)
        model_module = importlib.import_module(module_name)
        model_class = getattr(model_module, class_name)

        base_model=model_class.from_pretrained(self.model_name, 
            torch_dtype=torch.float16, 
            device_map="auto")

        if self.lora_path:
            self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        else:
            self.model = base_model

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        utils_module = importlib.import_module(config["utils_module"])
        self.process_vision_info = getattr(utils_module, "process_vision_info")

    def run_inference_on_image(_message: Dict[str,Any], image: Image.Image, image_name: str)->str:
        messages = [_message]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1582)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            print(output_text[0])
        return output_text[0]


    
