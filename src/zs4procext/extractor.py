import json
from typing import Any, Dict, List, Optional

import importlib_resources
from pydantic import BaseModel, PrivateAttr

from zs4procext.actions import (
    ACTION_REGISTRY,
    AQUEOUS_REGISTRY,
    FILTRATE_REGISTRY,
    MICROWAVE_REGISTRY,
    ORGANIC_REGISTRY,
    PH_REGISTRY,
    PISTACHIO_ACTION_REGISTRY,
    PRECIPITATE_REGISTRY,
    Add,
    CollectLayer,
    Filter,
    MakeSolution,
    Quench,
    SetTemperature,
)
from zs4procext.llm import ModelLLM
from zs4procext.parser import (
    ActionsParser,
    KeywordSearching,
    ParametersParser,
    SchemaParser,
)
from zs4procext.prompt import PromptFormatter


class ActionExtractorFromText(BaseModel):
    actions_type: str = "All"
    action_prompt_structure_path: Optional[str] = None
    chemical_prompt_structure_path: Optional[str] = None
    action_prompt_schema_path: Optional[str] = None
    chemical_prompt_schema_path: Optional[str] = None
    llm_model_name: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    _action_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)
    _action_parser: Optional[ActionsParser] = PrivateAttr(default=None)
    _condition_parser: Optional[ParametersParser] = PrivateAttr(default=None)
    _quantity_parser: Optional[ParametersParser] = PrivateAttr(default=None)
    _schema_parser: Optional[SchemaParser] = PrivateAttr(default=None)
    _filtrate_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _precipitate_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _aqueous_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _organic_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _microwave_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _ph_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _action_dict: Dict[str, Any] = PrivateAttr(default=ACTION_REGISTRY)

    def model_post_init(self, __context: Any) -> None:
        if self.action_prompt_schema_path is None:
            self.action_prompt_schema_path = str(
                importlib_resources.files("zs4procext")
                / "resources"
                / "organic_synthesis_actions_schema.json"
            )
        with open(self.action_prompt_schema_path, "r") as f:
            action_prompt_dict = json.load(f)
        self._action_prompt = PromptFormatter(**action_prompt_dict)
        self._action_prompt.model_post_init(self.action_prompt_structure_path)
        if self.chemical_prompt_schema_path is None:
            self.chemical_prompt_schema_path = str(
                importlib_resources.files("zs4procext")
                / "resources"
                / "chemicals_from_actions_schema.json"
            )
        with open(self.chemical_prompt_schema_path, "r") as f:
            chemical_prompt_dict = json.load(f)
        self._chemical_prompt = PromptFormatter(**chemical_prompt_dict)
        self._chemical_prompt.model_post_init(self.chemical_prompt_structure_path)
        if self.llm_model_parameters_path is None:
            llm_param_path = str(
                importlib_resources.files("zs4procext")
                / "resources"
                / "vllm_default_params.json"
            )
        else:
            llm_param_path = self.llm_model_parameters_path
        if self.llm_model_name is None:
            self._llm_model = ModelLLM(model_name="Llama2-70B-chat-hf")
        else:
            self._llm_model = ModelLLM(model_name=self.llm_model_name)
        if self.actions_type == "pistachio":
            self._action_dict = PISTACHIO_ACTION_REGISTRY
            self._ph_parser = KeywordSearching(keywords_list=["&^%#@&#@(*)"])
            self._ph_parser.model_post_init(None)
        else:
            self._ph_parser = KeywordSearching(keywords_list=PH_REGISTRY)
        self._ph_parser.model_post_init(None)
        self._llm_model.load_model_parameters(llm_param_path)
        self._llm_model.vllm_load_model()
        self._action_parser = ActionsParser()
        self._action_parser.model_post_init(None)
        self._condition_parser = ParametersParser(convert_units=False, amount=False)
        self._condition_parser.model_post_init(None)
        self._quantity_parser = ParametersParser(
            convert_units=False,
            time=False,
            temperature=False,
            pressure=False,
            atmosphere=False,
        )
        self._quantity_parser.model_post_init(None)
        atributes = ["chemical", "dropwise"]
        self._schema_parser = SchemaParser(atributes_list=atributes)
        self._schema_parser.model_post_init(None)
        self._filtrate_parser = KeywordSearching(keywords_list=FILTRATE_REGISTRY)
        self._filtrate_parser.model_post_init(None)
        self._precipitate_parser = KeywordSearching(keywords_list=PRECIPITATE_REGISTRY)
        self._precipitate_parser.model_post_init(None)
        self._aqueous_parser = KeywordSearching(keywords_list=AQUEOUS_REGISTRY)
        self._aqueous_parser.model_post_init(None)
        self._organic_parser = KeywordSearching(keywords_list=ORGANIC_REGISTRY)
        self._organic_parser.model_post_init(None)
        self._microwave_parser = KeywordSearching(keywords_list=MICROWAVE_REGISTRY)
        self._microwave_parser.model_post_init(None)

    @staticmethod
    def empty_action(action: Dict[str, Any]):
        content: Dict[str, Any] = action["content"]
        list_of_keys = content.keys()
        is_empty = True
        ignore_set = {"dropwise", "repetitions"}
        # chemical_set = {"material", "solvent", "material_1", "material_2"}
        for key in list_of_keys:
            if key == "meterials":
                if content[key] != []:
                    is_empty = False
                    break
            elif key in ignore_set:
                pass
            else:
                if content[key] is not None:
                    is_empty = False
                    break
        return is_empty

    @staticmethod
    def eliminate_empty_sequence(action_list: List[Dict[str, Any]], threshold: int):
        ignore_set = {
            "CollectLayer",
            "Concentrate",
            "Filter",
            "PhaseSeparation",
            "Purify",
        }
        empty_sequence = 0
        i = 0
        for action in action_list:
            if action["action"] in ignore_set:
                empty_sequence = 0
            elif ActionExtractorFromText.empty_action(action):
                empty_sequence = empty_sequence + 1
                if empty_sequence == threshold:
                    action_list = (
                        action_list[: i - threshold + 1] + action_list[i + 1 :]
                    )
                    i = i - threshold
                elif empty_sequence > threshold:
                    del action_list[i]
                    i = i - 1
            else:
                empty_sequence = 0
            i = i + 1
        return action_list

    def retrieve_actions_from_text(
        self, paragraph: str, stop_words: List[str]
    ) -> List[Any]:
        if (
            self._llm_model is None
            or self._chemical_prompt is None
            or self._action_prompt is None
            or self._action_parser is None
            or self._schema_parser is None
            or self._quantity_parser is None
            or self._condition_parser is None
            or self._ph_parser is None
            or self._microwave_parser is None
        ):
            raise AttributeError("You need to post initilize the class")
        action_prompt: str = self._action_prompt.format_prompt(paragraph)
        actions_response: str = self._llm_model.run_single_prompt(action_prompt)
        actions_info: Dict[str, List[str]] = self._action_parser.parse(actions_response)
        i = 0
        action_list: List = []
        for action_name in actions_info["actions"]:
            context = actions_info["content"][i]
            try:
                action = self._action_dict[action_name.lower()]
            except KeyError:
                action = None
            if action is None:
                print(action_name)
                if action_name.lower() in stop_words:
                    break
            elif action is SetTemperature:
                new_action: List[Dict[str, Any]] = action.generate_action(
                    context, self._condition_parser, self._microwave_parser
                )
                action_list.extend(new_action)
            elif action is MakeSolution or action is Add or action is Quench:
                chemical_prompt = self._chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._condition_parser,
                    self._ph_parser,
                )
                action_list.extend(new_action)
            elif action.type == "onlyconditions":
                new_action = action.generate_action(context, self._condition_parser)
                action_list.extend(new_action)
            elif action.type == "onlychemicals":
                chemical_prompt = self._chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context, schemas, self._schema_parser, self._quantity_parser
                )
                action_list.extend(new_action)
            elif action.type == "chemicalsandconditions":
                chemical_prompt = self._chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._condition_parser,
                )
                action_list.extend(new_action)
            elif action is Filter:
                new_action = action.generate_action(
                    context, self._filtrate_parser, self._precipitate_parser
                )
                action_list.extend(new_action)
            elif action is CollectLayer:
                new_action = action.generate_action(
                    context, self._aqueous_parser, self._organic_parser
                )
                action_list.extend(new_action)
            elif action.type is None:
                new_action = action.generate_action(context)
                action_list.extend(new_action)
            i = i + 1
        return ActionExtractorFromText.eliminate_empty_sequence(action_list, 5)
