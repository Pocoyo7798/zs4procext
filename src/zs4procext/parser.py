import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import importlib_resources
from pint import UnitRegistry
from pydantic import BaseModel, PrivateAttr, validator
from quantulum3 import parser
from trieregex import TrieRegEx as TRE


class Amount(BaseModel):
    value: List[str] = []
    repetitions: List[int] = []

class ComplexConditions(BaseModel):
    stirring_speed: List[str] = []
    heat_ramp: List[str] = []
    flow_rate: List[str] = []

class Conditions(BaseModel):
    duration: List[str] = []
    temperature: List[str] = []
    pressure: List[str] = []
    atmosphere: List[str] = []
    size: List[str] = []
    amount: Dict[str, List[Any]] = {}
    other: List[str] = []


class Parameters(BaseModel):
    time_units: List[str] = []
    temperature_units: List[str] = []
    pressure_units: List[str] = []
    quantity_units: List[str] = []
    size_units: List[str] = []
    time_words: List[str] = []
    temperature_words: List[str] = []
    pressure_words: List[str] = []
    atmosphere_words: List[str] = []
    amount_words: List[str] = []
    size_words: List[str] = []

class ComplexParameters(BaseModel):
    stirring_units: List[str] = []
    heat_ramp_units: List[str] = []
    flow_rate_units: List[str] = []


class ParametersParser(BaseModel):
    parser_params_path: str = str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "synthesis_parsing_parameters.json"
    )
    time: bool = True
    temperature: bool = True
    pressure: bool = True
    atmosphere: bool = True
    amount: bool = True
    size: bool = True
    convert_units: bool = True
    base_time: str = "minute"
    base_temperature: str = "degree_Celsius"
    base_pressure: str = "bar"
    base_volume: str = "milliliter"
    base_mass: str = "milligram"
    base_quantity: str = "millimole"
    base_size: str = "meter"
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _ureg: Optional[UnitRegistry] = PrivateAttr(default=None)
    _Q: Any = PrivateAttr(default=None)
    UnitRegistry()

    @validator("parser_params_path")
    def verify_path(cls, v) -> str:
        if Path(v).exists() is False:
            raise ValueError("The path given does not exist")
        return v

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        self._ureg = UnitRegistry()
        self._Q = self._ureg.Quantity
        with open(self.parser_params_path, "r") as f:
            parser_params_dict = json.load(f)
        parser_params = Parameters(**parser_params_dict)
        units_list: List[str] = []
        time_word_list: List[str] = [
            "$%##&#$@%"
        ]  # This avoids the problem of empty lists on regex
        temperature_word_list: List[str] = ["$%##&#$@%"]
        pressure_word_list: List[str] = ["$%##&#$@%"]
        atmosphere_word_list: List[str] = ["$%##&#$@%"]
        amount_word_list: List[str] = ["$%##&#$@%"]
        size_word_list: List[str] = ["$%##&#$@%"]
        if self.time is True:
            units_list += parser_params.time_units
            if parser_params.time_words != []:
                time_word_list = parser_params.time_words
        if self.temperature is True:
            units_list += parser_params.temperature_units
            if parser_params.temperature_words != []:
                temperature_word_list = parser_params.temperature_words
        if self.pressure is True:
            units_list += parser_params.pressure_units
            if parser_params.pressure_words != []:
                pressure_word_list = parser_params.pressure_words
        if self.atmosphere is True:
            if parser_params.atmosphere_words != []:
                atmosphere_word_list = parser_params.atmosphere_words
        if self.amount is True:
            units_list += parser_params.quantity_units
            if parser_params.amount_words != []:
                amount_word_list = parser_params.amount_words
        if self.size is True:
            units_list += parser_params.size_units
            if parser_params.size_words != []:
                size_word_list = parser_params.size_words
        units_tre: TRE = TRE(*units_list)
        time_word_tre: TRE = TRE(*time_word_list)
        temperature_word_tre: TRE = TRE(*temperature_word_list)
        pressure_word_tre: TRE = TRE(*pressure_word_list)
        atmosphere_word_tre: TRE = TRE(*atmosphere_word_list)
        amount_word_tre: TRE = TRE(*amount_word_list)
        size_word_tre: TRE = TRE(*size_word_list)
        regex: str = rf"([\"'\(\[\s,]((?P<repetitions1>\d+\.?,?\d*)[xX×]+)?(?P<number1>\+?-?-?\d+\.?,?\d*)(?P<unit1>.?)-*(?P<number2>\d*\.?,?\d*)\s*(?P<unit2>.?\s*{units_tre.regex()})([xX×]+(?P<repetitions2>\d+\.?,?\d*))?(?=[\)\]\s,\"'\(\.])|\b(?P<word>(?P<time>{time_word_tre.regex()})|(?P<temperature>{temperature_word_tre.regex()})|(?P<pressure>{pressure_word_tre.regex()})|(?P<atmosphere>{atmosphere_word_tre.regex()})|(?P<amount>{amount_word_tre.regex()})|(?P<size>{size_word_tre.regex()}))\b)"
        self._regex = re.compile(regex, re.IGNORECASE | re.MULTILINE)

    def transform_value(self, number: str, unit: str) -> tuple[str, str, str]:
        """divides a parameter into digits and units, and convert it into the desired units

        Args:
            parameter: pint object

        Returns:
            the digit, unit and type of the desired parameter
        """
        try:
            parameter: UnitRegistry.Quantity = self._Q(float(number), unit)
            unit_type: str
            if parameter.check("[time]") is True:
                unit_type = "duration"
                if self.base_time is not None and self.convert_units is True:
                    parameter.ito(self.base_time)
            elif parameter.check("[temperature]") is True:
                unit_type = "temperature"
                if self.base_temperature is not None and self.convert_units is True:
                    parameter.ito(self.base_temperature)
            elif parameter.check("[pressure]") is True:
                unit_type = "pressure"
                if self.base_pressure is not None and self.convert_units is True:
                    parameter.ito(self.base_pressure)
            elif parameter.check("[mass]") is True:
                unit_type = "quantity"
                if self.base_mass is not None and self.convert_units is True:
                    parameter.ito(self.base_mass)
            elif parameter.check("[length]") is True:
                unit_type = "size"
                if self.base_mass is not None and self.convert_units is True:
                    parameter.ito(self.base_mass)
            elif parameter.check("[substance]") is True:
                unit_type = "quantity"
                if self.base_quantity is not None and self.convert_units is True:
                    parameter.ito(self.base_quantity)
            elif parameter.check("[volume]") is True:
                unit_type = "quantity"
                if self.base_volume is not None and self.convert_units is True:
                    parameter.ito(self.base_volume)
            else:
                unit_type = "other"
            if self.convert_units is True:
                digit: str = str(parameter.magnitude)
                final_unit: str = str(parameter.units)
            else:
                digit = number
                final_unit = unit
        except Exception:
            unit_type = "other"
            digit = number
            final_unit = unit
        return digit, final_unit, unit_type

    def get_value(self, match: re.Match) -> Dict[str, Union[str, int]] | None:
        """get the value of a parsed result

        Args:
            value_parsed: tupple containing the parsed value

        Returns:
            a dictionary containing information about the parsed value
        """
        if match.group("number2") == "":
            value_type: str = "single"
            digit: str = match.group("number1")
            unit: str = match.group("unit1") + match.group("unit2")
            unit = unit.replace(" ", "")
            try:
                final_digit, final_unit, unit_type = self.transform_value(digit, unit)
            except Exception:
                return None
        else:
            value_type = "range"
            number1: str = match.group("number1")
            number2: str = match.group("number2")
            unit = match.group("unit2").replace(" ", "")
            try:
                digit1, final_unit, unit_type = self.transform_value(number1, unit)
                digit2, final_unit, unit_type = self.transform_value(number2, unit)
                final_digit = f"{digit1}-{digit2}"
            except Exception:
                return None
        if match.group("repetitions1") is not None:
            repetitions: int = round(float(match.group("repetitions1")))
        elif match.group("repetitions2") is not None:
            repetitions = round(float(match.group("repetitions2")))
        else:
            repetitions = 1
        if unit_type == "other":
            final_unit = unit
        return {
            "value_type": value_type,
            "number": final_digit,
            "condition_type": unit_type,
            "unit": final_unit,
            "repetitions": repetitions,
        }

    def get_string(self, match: re.Match) -> Dict[str, str] | None:
        """Get a dictionary containing information about a parsed result

        Args:
            match: parsed result contanining the strin

        Returns:
            a dictionary containing information about a parsed result
        """
        if match.group("word") is None:
            return None
        string = match.group("word")
        value_type: str = "word"
        if match.group("time") is not None:
            word_type: str = "duration"
        elif match.group("temperature") is not None:
            word_type = "temperature"
        elif match.group("pressure") is not None:
            word_type = "pressure"
        elif match.group("atmosphere") is not None:
            word_type = "atmosphere"
        elif match.group("amount") is not None:
            word_type = "amount"
        else:
            word_type = "other"
        return {"value_type": value_type, "condition_type": word_type, "string": string}

    def get_parameters(self, text: str) -> Conditions:
        """get all the parameters wanted from a string

        Args:
            text: string to be processed

        Raises:
            ValueError: if the parser was not initilized

        Returns:
            a list of all the parameters found
        """
        if self._regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.model_post_init(None)"
            )
        text = " " + text + " "  # needed to avoid errors at the regex parser
        results: Iterator[re.Match[str]] = self._regex.finditer(text)
        conditions: Conditions = Conditions()
        amount: Amount = Amount()
        condition: Optional[Dict[str, Any]]
        value: str
        for result in results:
            if result.group("number1") is not None:
                condition = self.get_value(result)
                if condition is not None:
                    condition_type = condition["condition_type"]
                    value = f"{condition['number']} {condition['unit']}"
                    repetitions = condition["repetitions"]
            else:
                condition = self.get_string(result)
                if condition is not None:
                    condition_type = condition["condition_type"]
                    value = condition["string"]
                    repetitions = "1"
            if condition is None:
                pass
            elif condition_type == "duration":
                conditions.duration.append(value)  # type: ignore
            elif condition_type == "temperature":
                conditions.temperature.append(value)  # type: ignore
            elif condition_type == "pressure":
                conditions.pressure.append(value)  # type: ignore
            elif condition_type == "atmosphere":
                conditions.atmosphere.append(value)  # type: ignore
            elif condition_type == "quantity":
                amount.value.append(value)  # type: ignore
                amount.repetitions.append(repetitions)  # type: ignore
            elif condition_type == "size":
                conditions.size.append(value)  # type: ignore
            else:
                conditions.other.append(value)  # type: ignore
        conditions.amount = amount.__dict__
        return conditions


class ComplexParametersParser(BaseModel):
    parser_params_path: str = str(
        importlib_resources.files("zs4procext")
        / "resources"
        / "synthesis_parsing_complex_parameters.json"
    )
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        with open(self.parser_params_path, "r") as f:
            parser_params_dict = json.load(f)
        parser_params = ComplexParameters(**parser_params_dict)
        stirring_units_list: List[str] =  parser_params.stirring_units
        heating_ramp_units_list: List[str] = parser_params.heat_ramp_units
        flow_rate_units_list: List[str] = parser_params.flow_rate_units
        stirring_units_tre: TRE = TRE(*stirring_units_list)
        heating_ramp_units_tre: TRE = TRE(*heating_ramp_units_list)
        flow_rate_units_tre: TRE = TRE(*flow_rate_units_list)
        regex: str = rf"([\"'\(\[\s,](?P<number1>\+?-?-?\d+\.?,?\d*)-*(?P<number2>\d*\.?,?\d*)\s*((?P<stirring_speed>[^-\),\[\]\d\s]?\s*{stirring_units_tre.regex()})|(?P<heat_ramp>.?\s*{heating_ramp_units_tre.regex()})|(?P<flow_rate>[^-\),\[\]\d\s]?\s*{flow_rate_units_tre.regex()}))(?=[\)\]\s,\"'\(\.]))"
        self._regex = re.compile(regex, re.IGNORECASE | re.MULTILINE)

    def generate_value(self, match: re.Match) -> Dict[str, str]:
        if match.group("number2") != "":
            value: str = f"{match.group('number1')}-{match.group('number2')}"
        else:
            value = f"{match.group('number1')}"
        if match.group("stirring_speed") is not None:
            unit: str = f"{match.group('stirring_speed')}"
            condition_type: str = "stirring_speed"
        if match.group("heat_ramp") is not None:
            unit = f"{match.group('heat_ramp')}"
            condition_type = "heat_ramp"
        if match.group("flow_rate") is not None:
            unit = f"{match.group('flow_rate')}"
            condition_type = "flow_rate"
        return {"value": f"{value} {unit}", "condition_type": condition_type}
        


    def get_parameters(self, text:str) -> ComplexConditions:
        if self._regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.model_post_init(None)"
            )
        text = " " + text + " "  # needed to avoid errors at the regex parser
        results: Iterator[re.Match[str]] = self._regex.finditer(text)
        conditions: ComplexConditions = ComplexConditions()
        for result in results:
            result_dict: Dict[str, str] = self.generate_value(result)
            condition_type: str = result_dict["condition_type"]
            value: str = result_dict["value"]
            if condition_type == "stirring_speed":
                conditions.stirring_speed.append(value)  # type: ignore
            elif condition_type == "heat_ramp":
                conditions.heat_ramp.append(value)  # type: ignore
            elif condition_type == "flow_rate":
                conditions.flow_rate.append(value)  # type: ignore
        return conditions

class ActionsParser(BaseModel):
    separators: List[str] = []
    type: str = "materials"
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        if self.type == "materials":
            self.separators = MATERIAL_SEPARATORS_REGISTRY
        elif self.type == "pistachio":
            self.separators = PISTACHIO_SEPARATORS_REGISTRY
        tre: TRE = TRE(*self.separators)
        self._regex = re.compile(f"\\b{tre.regex()}\\b", re.IGNORECASE | re.MULTILINE)

    def change_separators(
        self,
        extra_separators: List | None = None,
        separators_to_remove: List | None = None,
    ) -> None:
        """Change the separators from the parser

        Args:
            extra_separators: list of extra separator to consider. Defaults to None.
            separators_to_remove: list of separator to not be consider. Defaults to None.
        """
        tre: TRE = TRE(*self.separators)
        if extra_separators is not None:
            tre.add(*extra_separators)
        if separators_to_remove is not None:
            tre.remove(*separators_to_remove)
        self._regex = re.compile(f"\\b{tre.regex()}\\b", re.IGNORECASE | re.MULTILINE)

    def parse(self, text: str) -> Dict[str, List[str]]:
        """Extracts the different actions and content from a string

        Args:
            text: string to be parsed.

        Returns:
            A dictionary containing the action recognized and the content inside them
        """
        if self._regex is None:
            raise ValueError(
                "The regex was not initialize, initialize it by <object_name>.initialize()"
            )
        actions: List[str] = self._regex.findall(text)
        content: List[str] = self._regex.split(text)[1:]
        i = 0
        for action in actions[:-1]:
            if action == actions[i + 1] and len(content[i] < 6):
                del actions[i]
                del content[i]
            else:
                i += 1
        return {"actions": actions, "content": content}


class KeywordSearching(BaseModel):
    keywords_list: List[str]
    _regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)

    @validator("keywords_list")
    def list_not_empty(cls, v) -> List[str]:
        if v == []:
            raise ValueError(
                "The keyword list is empty, please give a valid keyword list"
            )
        return v

    def model_post_init(self, __context: Any) -> None:
        """initialize the parser object by compiling a regex code"""
        tre: TRE = TRE(*self.keywords_list)
        self._regex = re.compile(f"\\b{tre.regex()}\\b", re.IGNORECASE | re.MULTILINE)

    def find_keywords(self, text: str) -> List[str]:
        """find all the keywords inside a string

        Args:
            text: string to analyse

        Returns:
            a list containing all the keywords found
        """
        if self._regex is None:
            raise AttributeError("There is no valid regex loaded")
        return self._regex.findall(text)


class SchemaParser(BaseModel):
    atributes_list: List[str]
    limiters: Dict[str, str] = {"initial": "{", "final": "}"}
    _schema_regex: Optional[re.Pattern[str]] = PrivateAttr(default=None)
    _atributes_regex: Dict[str, re.Pattern[str]] = PrivateAttr(default={})

    @validator("limiters")
    def limiters_list_not_empty(cls, v) -> Dict[str, str]:
        if v == {}:
            raise ValueError(
                "The limiters dictionary is empty, please give a valid limiter dicionary. The correct format is:\n {{'inital' : 'put_inital_limiter_here', 'final' : 'put_final_limiter_here'}}"
            )
        elif "initial" not in v.keys() and "final" not in v.keys():
            raise ValueError(
                "The limiters format is not valid. The correct format is:\n {{'initial' : 'put_inital_limiter_here', 'final' : 'put_final_limiter_here'}}"
            )
        return v

    @validator("atributes_list")
    def atributes_list_not_empty(cls, v) -> List[str]:
        if v == []:
            raise ValueError("The atributes_list should not be an empty list")
        return v

    def model_post_init(self, _context: Any) -> None:
        limiters_list: List[str] = [self.limiters["initial"], self.limiters["final"]]
        tre: TRE = TRE(*limiters_list)
        self._schema_regex = re.compile(f"{tre.regex()}", re.IGNORECASE | re.MULTILINE)
        for atribute in self.atributes_list:
            self._atributes_regex[atribute] = re.compile(
                rf"[\"']*{atribute}[\"']*\s*[:=-]\s*[\"']*([^\"'{self.limiters['initial']}{self.limiters['final']}]*)[\"']*,*",
                re.IGNORECASE | re.MULTILINE,
            )

    def parse_schema(self, text: str) -> List[str]:
        """Obtain all schema results from a string

        Args:
            text: string containing the text to be parsed

        Raises:
            AttributeError: The regex parser was not compiled

        Returns:
            List contaning all schema results
        """
        if self._schema_regex is None:
            raise AttributeError(
                "You need to Initialize the object first using <object>.model_post_init method"
            )
        limiters: List[str] = self._schema_regex.findall(text)
        context: List[str] = self._schema_regex.split(text)
        if len(context) > len(limiters):
            context = context[1:]
        i = 0
        close_limiter_value: int = 0
        result: List[str] = []
        summed_context: str = ""
        for limiter in limiters:
            if limiter == self.limiters["initial"]:
                close_limiter_value += 1
            else:
                close_limiter_value -= 1
            if close_limiter_value <= 0:
                summed_context = summed_context + limiter
                result.append(summed_context)
                close_limiter_value = 0
                summed_context = ""
            else:
                summed_context = summed_context + limiter + context[i]
            i += 1
        return result

    def get_atribute_value(self, text: str, atribute: str):
        if atribute not in self.atributes_list:
            raise ValueError(
                f"The give atribute is not valid, the valid atributes are {self.atributes_list}"
            )
        regex: re.Pattern[str] = self._atributes_regex[atribute]
        result: List[Any] = regex.findall(text)
        return result


class DimensionlessParser:
    @classmethod
    def get_dimensionless_numbers(cls, context: str) -> List[str]:
        quants: List[Any] = parser.parse(context)
        dimensionless_list: List[str] = []
        for quant in quants:
            if quant.unit.entity.name == "dimensionless":
                dimensionless_list.append(str(quant.value))
        return dimensionless_list
    
PISTACHIO_SEPARATORS_REGISTRY: List[str] = [
        "Initialization",
        "Add",
        "Cool",
        "Heat",
        "SetTemperature",
        "Stir",
        "Concentrate",
        "Evaporate",
        "DrySolution",
        "Dry",
        "CollectLayer",
        "Collect",
        "Extract",
        "Wash",
        "MakeSolution",
        "Filter",
        "Recrystallize",
        "Crystallize",
        "Recrystalize",
        "Purify",
        "Quench",
        "PhaseSeparation",
        "AdjustPH",
        "Reflux",
        "DrySolid",
        "Degas",
        "Partition",
        "Sonicate",
        "Triturate",
        "FinalProduct",
        "Wait",
        "Note",
        "Notes",
        "FollowOtherProcedure",
        "NMR",
        "ESIMS",
        "Pour",
        "Distill",
        "Collect",
        "Dissolve",
        "Final Product:",
        "Remove",
        "Warm",
        "Dilute",
        "Solidify",
        "Provide",
        "Afford",
        "Obtain",
    ]

MATERIAL_SEPARATORS_REGISTRY: List[str] = [
        "Initialization",
        "Note",
        "Notes",
        "NMR",
        "ESIMS",
        "Add",
        "NewSolution",
        "Crystallization",
        "Separate",
        "Wash",
        "Wait",
        "Dry",
        "Calcination",
        "Stir",
        "IonExchange",
        "Repeat",
        "Cool",
        "Heat",
        "Grind",
        "Sieve",
        "AlkalineTreatment",
        "AcidTreatment"
    ]