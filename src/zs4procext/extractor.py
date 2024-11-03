import json
from typing import Any, Dict, Iterator, List, Optional
import os
import re

import importlib_resources
from pydantic import BaseModel, PrivateAttr, validator

from zs4procext.actions import (
    ACTION_REGISTRY,
    AQUEOUS_REGISTRY,
    BANNED_CHEMICALS_REGISTRY,
    CENTRIFUGATION_REGISTRY,
    EVAPORATION_REGISTRY,
    FILTER_REGISTRY,
    FILTRATE_REGISTRY,
    MATERIAL_ACTION_REGISTRY,
    MICROWAVE_REGISTRY,
    ORGANIC_REGISTRY,
    PH_REGISTRY,
    PISTACHIO_ACTION_REGISTRY,
    PRECIPITATE_REGISTRY,
    SAC_ACTION_REGISTRY,
    Add,
    AddMaterials,
    Cool,
    Crystallization,
    ChangeTemperature,
    ChangeTemperatureSAC,
    CoolSAC,
    CollectLayer,
    Filter,
    MakeSolution,
    NewSolution,
    Quench,
    Separate,
    SetTemperature,
    SonicateMaterial,
    StirMaterial,
    ThermalTreatment,
    WashMaterial,
    WashSAC,
)
from zs4procext.llm import ModelLLM
from zs4procext.parser import (
    ActionsParser,
    ComplexParametersParser,
    EquationFinder,
    KeywordSearching,
    MolarRatioFinder,
    MOLAR_RATIO_REGISTRY,
    NumberFinder,
    ParametersParser,
    SchemaParser,
    VariableFinder
)
from zs4procext.prompt import PromptFormatter


class ActionExtractorFromText(BaseModel):
    actions_type: str = "All"
    action_prompt_structure_path: Optional[str] = None
    chemical_prompt_structure_path: Optional[str] = None
    action_prompt_schema_path: Optional[str] = None
    chemical_prompt_schema_path: Optional[str] = None
    wash_chemical_prompt_schema_path: Optional[str] = None
    add_chemical_prompt_schema_path: Optional[str] = None
    solution_chemical_prompt_schema_path: Optional[str] = None
    llm_model_name: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    elementar_actions: bool = False
    _action_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _wash_chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _add_chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _solution_chemical_prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)
    _action_parser: Optional[ActionsParser] = PrivateAttr(default=None)
    _condition_parser: Optional[ParametersParser] = PrivateAttr(default=None)
    _complex_parser: Optional[ComplexParametersParser] = PrivateAttr(default=None)
    _quantity_parser: Optional[ParametersParser] = PrivateAttr(default=None)
    _schema_parser: Optional[SchemaParser] = PrivateAttr(default=None)
    _filtrate_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _precipitate_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _filter_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _centri_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _evaporation_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _aqueous_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _organic_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _microwave_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _ph_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _banned_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _molar_ratio_parser: Optional[MolarRatioFinder] = PrivateAttr(default=None)
    _action_dict: Dict[str, Any] = PrivateAttr(default=ACTION_REGISTRY)

    def model_post_init(self, __context: Any) -> None:
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
        if self.wash_chemical_prompt_schema_path is None:
            self._wash_chemical_prompt = self._chemical_prompt
        else:
            with open(self.wash_chemical_prompt_schema_path, "r") as f:
                wash_chemical_prompt_dict = json.load(f)
            self._wash_chemical_prompt = PromptFormatter(**wash_chemical_prompt_dict)
            self._wash_chemical_prompt.model_post_init(self.chemical_prompt_structure_path)
        if self.add_chemical_prompt_schema_path is None:
            self._add_chemical_prompt = self._chemical_prompt
        else:
            with open(self.add_chemical_prompt_schema_path, "r") as f:
                add_chemical_prompt_dict = json.load(f)
            self._add_chemical_prompt = PromptFormatter(**add_chemical_prompt_dict)
            self._add_chemical_prompt.model_post_init(self.chemical_prompt_structure_path)
        if self.solution_chemical_prompt_schema_path is None:
            self._solution_chemical_prompt = self._chemical_prompt
        else:
            with open(self.solution_chemical_prompt_schema_path, "r") as f:
                solution_chemical_prompt_dict = json.load(f)
            self._solution_chemical_prompt = PromptFormatter(**solution_chemical_prompt_dict)
            self._solution_chemical_prompt.model_post_init(self.chemical_prompt_structure_path)
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
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("zs4procext")
                    / "resources"
                    / "organic_synthesis_actions_schema.json"
                )
            self._action_dict = PISTACHIO_ACTION_REGISTRY
            self._ph_parser = KeywordSearching(keywords_list=["&^%#@&#@(*)"])
            self._ph_parser.model_post_init(None)
            self._aqueous_parser = KeywordSearching(keywords_list=AQUEOUS_REGISTRY)
            self._aqueous_parser.model_post_init(None)
            self._organic_parser = KeywordSearching(keywords_list=ORGANIC_REGISTRY)
            self._organic_parser.model_post_init(None)
            atributes = ["name", "dropwise"]
        elif self.actions_type == "materials":
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("zs4procext")
                    / "resources"
                    / "material_synthesis_actions_schema.json"
                )
            self._action_dict = MATERIAL_ACTION_REGISTRY
            self._ph_parser = KeywordSearching(keywords_list=PH_REGISTRY)
            self._ph_parser.model_post_init(None)
            self._filter_parser = KeywordSearching(keywords_list=FILTER_REGISTRY)
            self._filter_parser.model_post_init(None)
            self._centri_parser = KeywordSearching(keywords_list=CENTRIFUGATION_REGISTRY)
            self._centri_parser.model_post_init(None)
            self._evaporation_parser = KeywordSearching(keywords_list=EVAPORATION_REGISTRY)
            self._evaporation_parser.model_post_init(None)
            self._complex_parser = ComplexParametersParser()
            self._complex_parser.model_post_init(None)
            atributes = ["type", "name", "dropwise", "concentration", "amount"]
        elif self.actions_type == "sac":
            if self.action_prompt_schema_path is None:
                self.action_prompt_schema_path = str(
                    importlib_resources.files("zs4procext")
                    / "resources"
                    / "material_synthesis_actions_schema.json"
                )
            self._action_dict = SAC_ACTION_REGISTRY
            self._ph_parser = KeywordSearching(keywords_list=PH_REGISTRY)
            self._ph_parser.model_post_init(None)
            self._filter_parser = KeywordSearching(keywords_list=FILTER_REGISTRY)
            self._filter_parser.model_post_init(None)
            self._centri_parser = KeywordSearching(keywords_list=CENTRIFUGATION_REGISTRY)
            self._centri_parser.model_post_init(None)
            self._evaporation_parser = KeywordSearching(keywords_list=EVAPORATION_REGISTRY)
            self._evaporation_parser.model_post_init(None)
            self._complex_parser = ComplexParametersParser()
            self._complex_parser.model_post_init(None)
            atributes = ["type", "name", "dropwise", "concentration", "amount"]
        with open(self.action_prompt_schema_path, "r") as f:
                action_prompt_dict = json.load(f)
        self._action_prompt = PromptFormatter(**action_prompt_dict)
        self._action_prompt.model_post_init(self.action_prompt_structure_path)
        self._llm_model.load_model_parameters(llm_param_path)
        self._llm_model.vllm_load_model()
        self._action_parser = ActionsParser(type=self.actions_type, separators=self._action_prompt._action_separators)
        self._action_parser.model_post_init(None)
        self._condition_parser = ParametersParser(convert_units=False, amount=False)
        self._condition_parser.model_post_init(None)
        self._quantity_parser = ParametersParser(
            convert_units=False,
            time=False,
            temperature=False,
            pressure=False,
            atmosphere=False,
            size=False
        )
        self._quantity_parser.model_post_init(None)
        self._schema_parser = SchemaParser(atributes_list=atributes)
        self._schema_parser.model_post_init(None)
        self._filtrate_parser = KeywordSearching(keywords_list=FILTRATE_REGISTRY)
        self._filtrate_parser.model_post_init(None)
        self._banned_parser = KeywordSearching(keywords_list=BANNED_CHEMICALS_REGISTRY)
        self._banned_parser.model_post_init(None)
        self._precipitate_parser = KeywordSearching(keywords_list=PRECIPITATE_REGISTRY)
        self._precipitate_parser.model_post_init(None)
        self._microwave_parser = KeywordSearching(keywords_list=MICROWAVE_REGISTRY)
        self._microwave_parser.model_post_init(None)
        self._molar_ratio_parser = MolarRatioFinder(chemicals_list=MOLAR_RATIO_REGISTRY)
        self._molar_ratio_parser.model_post_init(None)

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
    
    @staticmethod
    def correct_action_list(action_list: List[Dict[str, Any]], elementar_actions: bool=False):
        print(action_list)
        for action in action_list:
            print(action)
        i = 0
        temperature = None
        add_new_solution = True
        i_new_solution = 0
        if len(action_list) > 1:
            last_action: Dict[str, Any] = action_list[-1]
            second_last_action: Dict[str, Any] = action_list[-1]
            if last_action["action"] == "Wait" and second_last_action["action"] in set(["Dry", "Wait", "ThermalTreatment", "Wash", "Separate"]):
                del action_list[-1]
        while i < len(action_list):
            action = action_list[i]
            if action["action"] == "Add" and add_new_solution is True:
                add_new_solution = False
                action_list.insert(i_new_solution, NewSolution(action_name="NewSolution").zeolite_dict())
                i = i + 1
            elif action["action"] == "NewSolution":
                add_new_solution = False
                temperature = None
                if i == len(action_list) - 1:
                    del action_list[i]
                elif action_list[i + 1]["action"] not in set(["Add", "ChangeTemperature", "SetAtmosphere", "Repetition"]):
                    del action_list[i]
                else:
                    i = i + 1
            elif action["action"] == "Repeat":
                post_action = action_list[i + 1]
                amount = float(action["content"]["amount"])
                if post_action["action"] =="Repeat":
                    new_amount = float(post_action["content"]["amount"])
                    if amount > new_amount:
                        del action_list[i]
                    else:
                        del action_list[i + 1]
                else:
                    i += 1
            elif action["action"] == "ChangeTemperature":
                content = action["content"]
                if content["temperature"] in set(["heat", "cool"]):
                    temperature = content["temperature"]
                    i = i + 1
                elif content["temperature"] == temperature:
                    del action_list[i]
                else:
                    temperature = content["temperature"]
                    i = i + 1
            elif action["action"] in set(["Wash", "Separate"]):
                add_new_solution = True
                i_new_solution = i + 1
                i = i + 1
            elif action["action"] == "Crystallization":
                i_new_solution = i + 1
                content = action["content"]
                if content["temperature"] is not None:
                    temperature = content["temperature"]
                if elementar_actions is True:
                    content = action["content"]
                    new_actions = []
                    b = 2
                    if content["temperature"] is not None:
                        temp = {'action': 'ChangeTemperature', 'content': {'temperature': content["temperature"], 'microwave': content["microwave"], 'heat_ramp': None}}
                    else:
                        temp = {'action': 'ChangeTemperature', 'content': {'temperature': "Heat", 'microwave': content["microwave"], 'heat_ramp': None}}
                    new_actions.append(temp)
                    if content["pressure"] is not None:
                        atm = {'action': 'SetAtmosphere', 'content': {'atmosphere': None, 'pressure': content["pressure"], "flow_rate": None}}
                    else:
                        atm = {'action': 'SetAtmosphere', 'content': {'atmosphere': None, 'pressure': "autogeneous", "flow_rate": None}}
                    new_actions.append(atm)
                    if content["duration"] is not None:
                        if content["stirring_speed"] is not None:
                            stir = {'action': 'Stir', 'content': {'duration': content["duration"], 'stirring_speed': content["stirring_speed"]}}
                        else:
                            stir = {'action': 'Wait', 'content': {'duration': content["duration"]}}
                        new_actions.append(stir)
                        b += 1
                    action_list = action_list[:i] + new_actions + action_list[i + 1:]
                    i += b
                else:
                    i = i + 1
            elif action["action"] == "Dry":
                i_new_solution = i + 1
                content = action["content"]
                if content["temperature"] is not None:
                    temperature = content["temperature"]
                if elementar_actions is True:
                    content = action["content"]
                    new_actions = []
                    b = 1
                    if content["temperature"] is not None:
                        temp = {'action': 'ChangeTemperature', 'content': {'temperature': content["temperature"], 'microwave': None, 'heat_ramp': None}}
                    else:
                        temp = {'action': 'ChangeTemperature', 'content': {'temperature': "Heat", 'microwave': None, 'heat_ramp': None}}
                    new_actions.append(temp)
                    if content["atmosphere"] is not None:
                        atm = {'action': 'SetAtmosphere', 'content': {'atmosphere': content["atmosphere"], 'pressure': None, "flow_rate": None}}
                        new_actions.append(atm)
                        b += 1
                    if content["duration"] is not None:
                        stir = {'action': 'Wait', 'content': {'duration': content["duration"]}}
                        new_actions.append(stir)
                        b += 1
                    action_list = action_list[:i] + new_actions + action_list[i + 1:]
                    i += b
                else:
                    i = i + 1
            elif action["action"] == "ThermalTreatment":
                i_new_solution = i + 1
                content = action["content"]
                if content["temperature"] is not None:
                    temperature = content["temperature"]
                if elementar_actions is True:
                    content = action["content"]
                    new_actions = []
                    b = 1
                    if content["temperature"] is not None:
                        temp = {'action': 'ChangeTemperature', 'content': {'temperature': content["temperature"], 'microwave': None, 'heat_ramp': content["heat_ramp"]}}
                    else:
                        temp = {'action': 'ChangeTemperature', 'content': {'temperature': "Heat", 'microwave': None, 'heat_ramp': content["heat_ramp"]}}
                    new_actions.append(temp)
                    if content["atmosphere"] is not None or content["flow_rate"] is not None:
                        atm = {'action': 'SetAtmosphere', 'content': {'atmosphere': content["atmosphere"], 'pressure': None, "flow_rate": content["flow_rate"]}}
                        new_actions.append(atm)
                        b += 1
                    if content["duration"] is not None:
                        stir = {'action': 'Wait', 'content': {'duration': content["duration"]}}
                        new_actions.append(stir)
                        b += 1
                    action_list = action_list[:i] + new_actions + action_list[i + 1:]
                    i += b
                else:
                    i = i + 1
            else:
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
        paragraph = self._molar_ratio_parser.substitute(paragraph)
        action_prompt: str = self._action_prompt.format_prompt(paragraph)
        actions_response: str = self._llm_model.run_single_prompt(action_prompt).strip()
        actions_response = actions_response.replace("\x03C", "°C")
        actions_response = actions_response.replace("oC", "°C") 
        actions_response = actions_response.replace("8C", "°C")
        actions_response = actions_response.replace("1C", "°C")
        actions_response = actions_response.replace("0C", "°C")
        actions_response = actions_response.replace("∘C", "°C")
        actions_response = actions_response.replace("◦C", "°C")
        actions_response = actions_response.replace("ºC", "°C")
        actions_response = actions_response.replace("C", "°C")
        actions_response = actions_response.replace("C", "°C")
        actions_response = actions_response.replace("℃", "°C")
        actions_response = actions_response.replace( "\x03C", "°C")
        print(actions_response)
        actions_info: Dict[str, List[str]] = self._action_parser.parse(actions_response)
        i = 0
        action_list: List = []
        for action_name in actions_info["actions"]:
            context = action_name + actions_info["content"][i]
            try:
                action = self._action_dict[action_name.lower()]
            except KeyError:
                action = None
            if action is None:
                print(action_name)
                if action_name.lower() in stop_words:
                    break
            elif action in set([SetTemperature]):
                new_action: List[Dict[str, Any]] = action.generate_action(
                    context, self._condition_parser, self._microwave_parser
                )
                action_list.extend(new_action)
            elif action in set([ChangeTemperature, Crystallization, Cool, ChangeTemperatureSAC, CoolSAC]):
                new_action: List[Dict[str, Any]] = action.generate_action(
                    context, self._condition_parser, self._complex_parser, self._microwave_parser
                )
                action_list.extend(new_action)
            elif action in set([ThermalTreatment, StirMaterial, SonicateMaterial]):
                new_action = action.generate_action(context, self._condition_parser, self._complex_parser)
                action_list.extend(new_action)
            elif action in set([MakeSolution, Add, Quench, AddMaterials, NewSolution]):
                if action is AddMaterials or action is Add:
                    chemical_prompt = self._add_chemical_prompt.format_prompt(context)
                elif action is NewSolution or action is MakeSolution:
                    chemical_prompt = self._solution_chemical_prompt.format_prompt(context)
                else:
                    chemical_prompt = self._chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt).strip()
                print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context,
                    schemas,
                    self._schema_parser,
                    self._quantity_parser,
                    self._condition_parser,
                    self._ph_parser,
                    self._banned_parser
                )
                action_list.extend(new_action)
            elif action is WashMaterial:
                chemical_prompt = self._wash_chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context, schemas, self._schema_parser, self._quantity_parser, self._centri_parser, self._filter_parser, self._ph_parser, self._banned_parser
                )
                action_list.extend(new_action)
            elif action is WashSAC:
                chemical_prompt = self._wash_chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context, schemas, self._schema_parser, self._quantity_parser, self._condition_parser, self._centri_parser, self._filter_parser, self._ph_parser, self._banned_parser
                )
                action_list.extend(new_action)
            elif action.type == "onlyconditions":
                new_action = action.generate_action(context, self._condition_parser)
                action_list.extend(new_action)
            elif action.type == "onlychemicals":
                chemical_prompt = self._chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                print(chemical_response)
                schemas = self._schema_parser.parse_schema(chemical_response)
                new_action = action.generate_action(
                    context, schemas, self._schema_parser, self._quantity_parser
                )
                action_list.extend(new_action)
            elif action.type == "chemicalsandconditions":
                chemical_prompt = self._chemical_prompt.format_prompt(context)
                chemical_response = self._llm_model.run_single_prompt(chemical_prompt)
                print(chemical_response)
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
            elif action is Separate:
                new_action = action.generate_action(
                    context, self._filtrate_parser, self._precipitate_parser,
                    self._centri_parser, self._filter_parser, self._evaporation_parser
                )
                action_list.extend(new_action)
            elif action.type is None:
                new_action = action.generate_action(context)
                action_list.extend(new_action)
            i = i + 1
        if self.actions_type == "pistachio":
            final_actions_list: List[Any] = ActionExtractorFromText.eliminate_empty_sequence(action_list, 5)
        elif self.actions_type == "materials":
            final_actions_list = ActionExtractorFromText.correct_action_list(action_list, elementar_actions=self.elementar_actions)
        else:
            final_actions_list = action_list

        return final_actions_list

class SamplesExtractorFromText(BaseModel):
    prompt_structure_path: Optional[str] = None
    prompt_schema_path: Optional[str] = None
    llm_model_name: Optional[str] = None
    llm_model_parameters_path: Optional[str] = None
    _schema_parser: Optional[SchemaParser] = PrivateAttr(default=None)
    _prompt: Optional[PromptFormatter] = PrivateAttr(default=None)
    _llm_model: Optional[ModelLLM] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.prompt_schema_path is None:
            self.prompt_schema_path = str(
                importlib_resources.files("zs4procext")
                / "resources"
                / "find_samples_procedures_schema.json"
            )
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
        if self.llm_model_name is None:
            self._llm_model = ModelLLM(model_name="Llama2-70B-chat-hf")
        else:
            self._llm_model = ModelLLM(model_name=self.llm_model_name)
        self._llm_model.load_model_parameters(llm_param_path)
        self._llm_model.vllm_load_model()
        atributes = ["name", "preparation", "yield"]
        self._schema_parser = SchemaParser(atributes_list=atributes)
        self._schema_parser.model_post_init(None)
    
    def retrieve_samples_from_text(self, paragraph: str) -> List[Any]:
        prompt: str = self._prompt.format_prompt(paragraph)
        response: str = self._llm_model.run_single_prompt(prompt)
        print(response)
        schemas: List[str] = self._schema_parser.parse_schema(response)
        samples_list: List[Any] = []
        i = 1
        for schema in schemas:
            sample_dict = {}
            name_list: List[str] = self._schema_parser.get_atribute_value(schema, "name")
            procedure_list: List[str] = self._schema_parser.get_atribute_value(schema, "preparation")
            yield_list: List[str] = self._schema_parser.get_atribute_value(schema, "yield")
            if len(name_list) > 0:
                sample_dict["sample"] = name_list[0]
            else:
                sample_dict["sample"] = f"sample {i}"
                i += 1
            if len(procedure_list) > 0:
                if procedure_list[0].strip().lower() == "n/a":
                    sample_dict["procedure"] = None
                else:
                    sample_dict["procedure"] = procedure_list[0]
            else:
                sample_dict["procedure"] = None
            if len(yield_list) > 0:
                if yield_list[0].strip().lower() == "n/a":
                    sample_dict["yield"] = None
                else:
                    sample_dict["yield"] = yield_list[0]
            else:
                sample_dict["yield"] = None
            samples_list.append(sample_dict)
        return samples_list
    
class MolarRatioExtractorFromText(BaseModel):
    chemicals_path: Optional[str] = None
    _finder: Optional[MolarRatioFinder] = PrivateAttr(default=None)
    _chemical_parser: Optional[KeywordSearching] = PrivateAttr(default=None)
    _number_parser: Optional[NumberFinder] = PrivateAttr(default=None)
    _variable_parser: VariableFinder = VariableFinder()
    _equation_parser: EquationFinder = EquationFinder()

    @validator("chemicals_path")
    def layer_options(cls, chemicals_path):
        if chemicals_path is None:
            pass
        elif os.path.splitext(chemicals_path)[-1] != ".txt":
            raise NameError("The file should be a .txt file")
        return chemicals_path

    def model_post_init(self, __context: Any) -> None:
        if self.chemicals_path is None:
            self._finder = MolarRatioFinder(chemicals_list=MOLAR_RATIO_REGISTRY)
            self._chemical_parser = KeywordSearching(keywords_list=MOLAR_RATIO_REGISTRY)
        else:
            with open(self.chemicals_path, "r") as f:
                chemicals_list = f.readlines()
            self._finder = MolarRatioFinder(chemicals_list=chemicals_list)
            self._chemical_parser = KeywordSearching(keywords_list=chemicals_list)
        self._finder.model_post_init(None)
        self._chemical_parser.model_post_init(False)
        self._number_parser = NumberFinder()
        self._number_parser.model_post_init(None)

    def correct_variables(self, ratio_dict: Dict[str, Optional[str]], text: str) -> tuple:
        keys: List[str] = list(ratio_dict.keys())
        conversion_dict: Dict[str, str] = {}
        for key in keys:
            ratio: Optional[str] = ratio_dict[key]
            if ratio is None:
                pass
            elif ratio.isnumeric():
                pass
            elif len(re.findall("[a-zA-Z]", ratio)) == 1:
                value_found: Optional[str] = self._variable_parser.find_value(ratio, text)
                conversion_dict[ratio] = key
                if value_found is None:
                    pass
                elif value_found[-1] == ".":
                    ratio = value_found[:-1]
                else:
                    ratio = value_found
                text = text.replace(ratio, "")
                ratio = ratio.strip()
                ratio = ratio.replace(", ", ",")
                ratio = ratio.replace(" and ", ",")
                ratio = ratio.replace(" ", ",")
            ratio_dict[key] = ratio
        return ratio_dict, conversion_dict, text

    def correct_ratios(self, ratio_dict: Dict[str, Optional[str]], text: str) -> tuple:
        keys: List[str] = list(ratio_dict.keys())
        numbers_list = self._number_parser.find_numbers_list(text, len(keys))
        if numbers_list is None:
            pass
        else:
            if numbers_list[-1] == ".":
                numbers_list = numbers_list[:-1]
            text = text.replace(numbers_list, "")
            numbers_list.replace("and", ",")
            values_list = re.split("[,:\/]", numbers_list)
            i = 0
            for key in keys:
                ratio_dict[key] = values_list[i]
                i += 1
        return ratio_dict, text
    
    def find_equations(self, text: str) -> tuple:
        equation_list: List[str] = self._equation_parser.find_all(text)
        for equation in equation_list:
            text = text.replace(equation, "")
        return equation_list, text
    
    def find_ratios(self, text: str) -> Dict[str, str]:
        molar_ratios: Dict[str, str] = {}
        all_ratios: Iterator[re.Match[str]] = self._finder.single_ratios(text)
        ratios_found: bool = False
        for ratio in all_ratios:
            print(ratio)
            ratios_found = True
            chemical1 = ratio.group("chemical1")
            chemical2 = ratio.group("chemical2")
            value = ratio.group("value")
            try:
                initial_value: str = molar_ratios[chemical2]
                list_initial_values: List[str] = re.split("[-–−]", initial_value)
            except KeyError:
                molar_ratios[chemical2] = "1"
                list_initial_values = ["1"]
            final_values: str = ""
            for initial_value in list_initial_values:
                new_value = float(value) * float(initial_value)
                final_values = final_values + f"{new_value}-"
            final_values = final_values[:-1]
            try:
                molar_ratios[chemical1] = molar_ratios[chemical1] + f",{final_values}"
            except KeyError:
                 molar_ratios[chemical1] = final_values
        if ratios_found is False:
            chemical_values: Iterator[re.Match[str]] = self._finder.single_values(text)
            for chemical in chemical_values:
                chemical_name: str = chemical.group("chemical")
                ratio_value: str = chemical.group("value")
                molar_ratios[chemical_name] = ratio_value
        return molar_ratios

    def extract_molar_ratio(self, text: str) -> List[Dict[str, Any]]:
        text = text.replace(" ­", "")
        molar_ratio_list: List[Any] = self._finder.find_molar_ratio(text)
        molar_ratios_result: List[Dict[str, Any]] = []
        equations: List[str] = []
        conversion_dict: Dict[str, str] = {}
        for molar_ratio in molar_ratio_list:
            string: str = molar_ratio[0]
            print(string)
            chemical_information: Dict[str, Any] = self._finder.find_chemical_information(string)
            ratio_dict: Dict[str,str] = chemical_information["result"]
            equations, text = self.find_equations(text)
            if chemical_information["values_found"] is False:
                ratio_dict, text =self.correct_ratios(ratio_dict, text)
            ratio_dict, conversion_dict, text =self.correct_variables(ratio_dict, text)
            if len(ratio_dict.keys()) > 2:
                molar_ratios_result.append(ratio_dict)
            else:
                conversion_dict = {}
        if molar_ratio_list == []:
            new_ratio_dict: Dict[str,str] = self.find_ratios(text)
            if len(new_ratio_dict.keys()) > 2:
                molar_ratios_result.append(new_ratio_dict)
        return {"molar_ratios": molar_ratios_result, "equations": equations, "letters": conversion_dict}
