from zs4procext.prompt import PromptFormatter
from zs4procext.llm import ModelLLM
from zs4procext.parser import ParametersParser
import json

with open("prompt_schema.json", "r") as f:
            prompt_dict = json.load(f)
text = "The alkaline treatment process was carried out using a 0.2 mol. L-1 NaOH solution on a parent zeolite calined at 673.15 K. In all experiments, 1 g of ZSM-5 zeolite, 100 mL of solution and heating at 338 K under reflux system were used. The duration of the process was limited to 30 and 10 min for conventional electric and microwave (500 W) heating’s, respectively."
prompt = PromptFormatter(**prompt_dict)
prompt.model_post_init("phi_mini_4k_template.json")
llm_model = ModelLLM("microsoft/Phi-3-mini-4k-instruct")
llm_model.load_model_parameters("model_parameters.json")
llm_model.vllm_load_model()
final_prompt = prompt.format_prompt(text)
response = llm_model.run_single_prompt(prompt).strip()
print(response)
temperature_parser = ParametersParser(
            parser_params_path = "parser_parameters.json",
            convert_units=True,
            time=False,
            temperature=True,
            pressure=False,
            amount=False,
            atmosphere=False,
            size=False
        )
temperature = temperature_parser.get_parameters(response)["temperature"]
print(temperature)