from typing import Any, Dict, List, Optional

import importlib_resources
from langchain.prompts import BasePromptTemplate, load_prompt
from pydantic import BaseModel, PrivateAttr


class PromptFormatter(BaseModel):
    expertise: str = ""
    initialization: str = ""
    objective: str = "No specific objective, just chatting..."
    actions: Dict[str, str] = {}
    answer_schema: Dict[str, str] = {}
    conclusion: str = ""
    _loaded_prompt: Optional[BasePromptTemplate] = PrivateAttr(default=None)
    _action_sperators: Optional[List[str]] = PrivateAttr(default=None)
    _answer_schema: Optional[str] = PrivateAttr(default=None)
    _action_list: Optional[str] = PrivateAttr(default=[None])

    def actions_to_string(
        self, action_intialization_key: str = "Initialization"
    ) -> str:
        """Generate a formatted action list

        Args:
            action_intialization_key: Key of the intialization text. Defaults to "Initialization".

        Raises:
            Warning: If the action_initializatio_key does not exist in the action dictionary

        Returns:
            a formatted action list
        """
        if self.actions == {}:
            return ""
        self._action_separators: List[str] = [
            action_key
            for action_key in self.actions.keys()
            if action_intialization_key not in action_key
        ]
        actions: List[str] = []
        if action_intialization_key not in self.actions:
            (
                "Warning: The actions were provided without a valid action_initializatio_key"
            )
        else:
            actions.append(f"{self.actions[action_intialization_key]}\n")
        for action in self._action_separators:
            actions.append(f"-'{action}' : {self.actions[action]}\n")
        return "".join(actions)

    def answer_schema_to_string(
        self, action_intialization_key: str = "Initialization"
    ) -> str:
        """Generate a formatted answer schema

        Args:
            action_intialization_key: Key of the intialization text. Defaults to "Initialization".

        Raises:
            Warning: If the action_initializatio_key does not exist in the action dictionary

        Returns:
            a formatted answer schema
        """
        if self.answer_schema == {}:
            return ""
        schema_list: List[str] = [
            schema_key
            for schema_key in self.answer_schema.keys()
            if action_intialization_key not in schema_key
        ]
        answer_schema_list: List[str] = []
        if action_intialization_key not in self.answer_schema:
            (
                "Warning: The answer schema was provided without a valid action_initialization_key"
            )
        else:
            answer_schema_list.append(
                f"{self.answer_schema[action_intialization_key]}\n"
            )
        for schema in schema_list:
            answer_schema_list.append(f"{self.answer_schema[schema]}\n")
        return "".join(answer_schema_list)

    def model_post_init(self, __context: Any) -> None:
        if __context is None:
            __context = str(
                importlib_resources.files("zs4procext")
                / "resources"
                / "organic_synthesis_actions_last_template.json"
            )
            loaded_prompt = load_prompt(__context)
            self._loaded_prompt = loaded_prompt
        else:
            loaded_prompt = load_prompt(__context)
            self._loaded_prompt = loaded_prompt
        action_list = self.actions_to_string()
        self._action_list = action_list
        answer_schema = self.answer_schema_to_string()
        self._answer_schema = answer_schema
        if self.expertise != "":
            self.expertise = self.expertise + "\n"
        if self.initialization != "":
            self.initialization = self.initialization + "\n"
        if self.objective != "":
            self.objective = self.objective + "\n"

    def format_prompt(
        self,
        context: str = "",
    ) -> str:
        """Generate a formatted prompt.

        Args:
            context: Text to be analysed. Defaults to "".
            prompt_template_path: Path to the prompt template file. Defaults to "".

        Returns:
            a formatted prompt.
        """
        if self._loaded_prompt is None:
            raise AttributeError(
                "There is no prompt loaded, you need to post init the object."
            )
        formatted_prompt: str = self._loaded_prompt.format(
            expertise=self.expertise,
            initialization=self.initialization,
            objective=self.objective,
            context=f"'{context}'\n",
            actions=self._action_list,
            answer_schema=self._answer_schema,
            conclusion=self.conclusion,
        )
        return formatted_prompt


TEMPLATE_REGISTRY: Dict[str, str] = {
    "meta-llama/Llama-2-7b-chat-hf": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "llama2_default_chat_template.json"
    ),
    "meta-llama/Llama-2-13b-chat-hf": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "llama2_default_chat_template.json"
    ),
    "meta-llama/Llama-2-70b-chat-hf": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "llama2_default_chat_template.json"
    ),
    "lmsys/vicuna-13b-v1.5-16k": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "vicuna_default_chat_template.json"
    ),
    "lmsys/vicuna-33b-v1.3": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "vicuna_default_chat_template.json"
    ),
    "mistralai/Mistral-7B-Instruct-v0.1": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mistral_instruct_default_template.json"
    ),
    "mistralai/Mistral-7B-v0.1": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mistral_default_template.json"
    ),
    "llm-agents/tora-13b-v1.0": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mistral_default_template.json"
    ),
    "llm-agents/tora-70b-v1.0": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mistral_default_template.json"
    ),
    "openlm-research/open_llama_7b": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "open_llama_default_template.json"
    ),
    "openlm-research/open_llama_13b": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "open_llama_default_template.json"
    ),
    "mosaicml/mpt-30b-chat": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mpt_default_chat_template.json"
    ),
    "mosaicml/mpt-30b": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mpt_default_template.json"
    ),
    "mistralai/Mistral-7B-Instruct-v0.2": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mistral_default_template.json"
    ),
    "openchat/openchat_3.5" : str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "open_chat_default_template.json"
    ),
    "openchat/openchat-3.5-0106" : str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "open_chat_default_template.json"
    ),
    "openchat/openchat-3.6-8b-20240522" : str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "open_chat_3.6_default_template.json"
    ),
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mistral_default_template.json"
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "mistral_instruct_default_template.json"
    ),
    "meta-llama/Meta-Llama-3-8B-Instruct": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "llama3_default_instruct_template.json"
    ),
    "microsoft/Phi-3-medium-4k-instruct": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "phi_default_instruct_template.json"
    ),
        "microsoft/Phi-3-medium-128k-instruct": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "phi_default_instruct_template.json"
    ),
    "microsoft/Phi-3-mini-4k-instruct": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "phi_default_instruct_template.json"
    ),
    "Qwen/Qwen1.5-7B-Chat": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "qwen1.5_default_chat_template.json"
    ),
    "berkeley-nest/Starling-LM-7B-alpha": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "starlinglm_default_template.json"
    ),
    "SanjiWatsuki/Kunoichi-DPO-v2-7B": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "kunoichi-dp_default_chat_template.json"
    ),
    "HuggingFaceH4/zephyr-7b-beta": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "zephyr_default_chat_template.json"
    ),
    "google/gemma-1.1-7b-it": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "gemma_default_it_template.json"
    ),
    "google/gemma-1.1-2b-it": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "gemma_default_it_template.json"
    ),
    "dreamgen/WizardLM-2-7B": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "wizardlm-2_default_template.json"
    ),
    "ibm-granite/granite-7b-instruct": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "ibm_granite_default_template.json"
    ),
    "ibm/merlinite-7b": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "ibm_granite_default_template.json"
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "llama31_default_instruct_template.json"
    )
    ,
    "upstage/SOLAR-10.7B-Instruct-v1.0": str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "solar_default_template.json"
    )
}