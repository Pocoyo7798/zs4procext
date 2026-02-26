import json
import torch
import importlib
from pydantic import BaseModel  

from xlmexlab.parser import ImageParser


class ModelWithAdapter(BaseModel):
    base_model_path: str
    imports_config_path: str
    adapter_path: str = None

    def __init__(self, **data):
        super().__init__(**data)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load imports configuration
        with open(self.imports_config_path) as f:
            self.imports_config = json.load(f)

        # Dynamically load classes and functions
        self.ModelClass = self._load_class(self.imports_config["model_class"])
        self.ProcessorClass = self._load_class(self.imports_config["processor_class"])
        self.process_vision_info = self._load_function(self.imports_config["vision_utils"])

        # Set default dtype
        torch_dtype = torch.float16

        # Load model
        self.model = self.ModelClass.from_pretrained(
            self.base_model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load processor
        self.processor = self.ProcessorClass.from_pretrained(self.base_model_path)

        # Load adapter if provided
        if self.adapter_path:
            self.model.load_adapter(self.adapter_path)

    def _load_class(self, dotted_path: str):
        module_name, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _load_function(self, dotted_path: str):
        module_name, func_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def generate(self, messages):
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process images / videos
        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        # Remove prompt tokens
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode text
        output = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parser.parse(output_text[0])
        parsed_output = parser.get_data_dict()

        return parsed_output
