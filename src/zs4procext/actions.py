from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, validator, PrivateAttr
import re

from zs4procext.parser import (
    Conditions,
    ComplexParametersParser,
    DimensionlessParser,
    KeywordSearching,
    ParametersParser,
    SchemaParser,
)


class Chemical(BaseModel):
    name: str = ""
    quantity: Optional[List[str]] = []

    def get_chemical(self, schema: str, schema_parser: SchemaParser) -> bool:
        """get the chemical name from a schema

        Args:
            schema: string containing the schema

        Returns:
            True if the chemical was added dropwise or Flase otherwise
        """
        chemical_list = schema_parser.get_atribute_value(schema, "name")
        dropwise_list = schema_parser.get_atribute_value(schema, "dropwise")
        if len(chemical_list) == 0:
            pass
        elif len(chemical_list) == 1:
            self.name = chemical_list[0]
        else:
            print("Warning: Two different chemical names have been found!")
            self.name = chemical_list[0]
        if len(dropwise_list) == 0:
            dropwise = "False"
        elif len(dropwise_list) == 1:
            dropwise = dropwise_list[0]
        else:
            print("Warning: Two different dropwise values have been found!")
            dropwise = dropwise_list[0]
        dropwise = dropwise.strip()
        if dropwise.lower() == "true":
            new_dropwise = True
        else:
            new_dropwise = False
        return new_dropwise


    def get_quantity(self, text: str, amount_parser: ParametersParser, get_concentration: bool=False) -> Any:
        """get the amount of a chemical inside a string

        Args:
            text: string to be analysed

        Raises:
            AttributeError: If the theres no ParameterParser loaded

        Returns:
            the amount of adding repetions of a chemical and concentration if asked for
        """
        amount: Conditions = amount_parser.get_parameters(text)
        amount_dict = amount.amount
        self.quantity = amount_dict["value"]  # type: ignore
        if len(amount_dict["repetitions"]) == 0:  # type: ignore
            max_repetitions: int = 1
        else:
            max_repetitions = int(max(amount_dict["repetitions"]))  # type: ignore
        if get_concentration is True:
            return amount.concentration, max_repetitions
        else:
            return max_repetitions


class ChemicalsMaterials(Chemical):
    concentration: Optional[List[str]] = []
    _chemical_type: str = PrivateAttr(default="reactant")
    def get_chemical_materials(self, schema: str, schema_parser: SchemaParser) -> bool:
        dropwise = self.get_chemical(schema, schema_parser)
        if len(schema_parser.get_atribute_value(schema, "type")) > 0:
            if schema_parser.get_atribute_value(schema, "type")[0].lower() == "final solution":
                self._chemical_type = "final solution"
        concentration_list: List[str] = schema_parser.get_atribute_value(schema, "concentration")
        if len(concentration_list) == 0:
            pass
        elif concentration_list[0].replace(",", "").strip().lower() == "n/a":
            pass
        else:
            self.concentration = concentration_list
        return dropwise

class ChemicalInfo(BaseModel):
    chemical_list: list[Chemical] = []
    dropwise: list[bool] = []
    final_solution: Optional[Chemical] = None
    repetitions: int = 1

class ChemicalInfoMaterials(ChemicalInfo):
    chemical_list: list[ChemicalsMaterials] = []
    final_solution: Optional[ChemicalsMaterials] = None


class Actions(BaseModel):
    action_name: str = ""
    action_context: str = ""
    type: ClassVar[Optional[str]] = None

    def transform_into_pistachio(self) -> Dict[str, Any]:
        action_name: str = self.action_name
        if type(self) is SetTemperature:
            action_dict = self.model_dump(
                exclude={"action_name", "action_context", "duration", "pressure"}
            )
        else:
            action_dict = self.model_dump(
                exclude={"action_name", "action_context", "pressure"}
            )
        return {"action": action_name, "content": action_dict}
    
    def zeolite_dict(self) -> Dict[str, Any]:
        action_name: str = self.action_name
        print(self)
        if type(self) in set([ChangeTemperature, Cool]):
            action_dict = self.model_dump(
                exclude={"action_name", "action_context", "pressure", "duration", "stirring_speed"}
            )
        elif type(self) in set([WaitMaterial, StirMaterial]):
            action_dict = self.model_dump(
                exclude={"action_name", "action_context", "temperature"}
            )
            print(action_dict)
        elif type(self) is AddMaterials:
            action_dict = self.model_dump(
                exclude={"action_name", "action_context", "atmosphere", "temperature"}
            )
        else:
            action_dict = self.model_dump(
                exclude={"action_name", "action_context"}
            )
        return {"action": action_name, "content": action_dict}

class ActionsWithchemicals(Actions):
    type: ClassVar[Optional[str]] = "onlychemicals"

    @classmethod
    def validate_chemicals(
        cls,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        context: str,
    ) -> ChemicalInfo:
        chemical_info = ChemicalInfo()
        repetitions_list: List[int] = []
        for schema in schemas:
            new_chemical: Chemical = Chemical()
            dropwise = new_chemical.get_chemical(schema, schema_parser)
            if len(schemas) > 1:
                repetitions = new_chemical.get_quantity(schema, amount_parser)
            else:
                repetitions = new_chemical.get_quantity(context, amount_parser)
            if new_chemical.name == "":
                pass
            elif new_chemical.name.strip().lower() == "n/a":
                pass
            else:
                chemical_info.chemical_list.append(new_chemical)
                chemical_info.dropwise.append(dropwise)
                repetitions_list.append(repetitions)
        if len(repetitions_list) == 0:
            chemical_info.repetitions = 1
        else:
            chemical_info.repetitions = max(repetitions_list)
        return chemical_info
    
    @classmethod
    def validate_chemicals_materials(
        cls,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        context: str,
    ) -> ChemicalInfo:
        chemical_info = ChemicalInfoMaterials()
        repetitions_list: List[int] = []
        for schema in schemas:
            new_chemical: ChemicalsMaterials = ChemicalsMaterials()
            dropwise = new_chemical.get_chemical_materials(schema, schema_parser)
            if len(schemas) > 1:
                repetitions = new_chemical.get_quantity(schema, amount_parser)
            else:
                repetitions = new_chemical.get_quantity(context, amount_parser)
            if new_chemical.name == "":
                pass
            elif new_chemical.name.strip().lower() == "n/a":
                pass
            elif new_chemical._chemical_type == "final solution":
                chemical_info.final_solution = new_chemical
            else:
                chemical_info.chemical_list.append(new_chemical)
                chemical_info.dropwise.append(dropwise)
                repetitions_list.append(repetitions)
        if len(repetitions_list) == 0:
            chemical_info.repetitions = 1
        else:
            chemical_info.repetitions = max(repetitions_list)
        return chemical_info

class ActionsWithConditons(Actions):
    type: ClassVar[Optional[str]] = "onlyconditions"

    def validate_conditions(self, conditions_parser: ParametersParser, complex_conditions_parser: Optional[ComplexParametersParser]=None, add_others: bool=False) -> None:
        conditions: Dict[str, Any] = conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        if complex_conditions_parser is not None:
            complex_conditions = complex_conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        for atribute in self.__dict__.keys():
            try:
                new_value = conditions[atribute][0]
            except Exception:
                if len(conditions["other"]) > 0 and add_others is True:
                    new_value = conditions["other"][0]
                else:
                    new_value = self.__dict__[atribute]
            if complex_conditions_parser is not None:
                try:
                    new_value = complex_conditions[atribute][0]
                except Exception:
                    pass
            setattr(self, atribute, new_value)


class ActionsWithChemicalAndConditions(Actions):
    type: ClassVar[Optional[str]] = "chemicalsandconditions"

    @classmethod
    def validate_chemicals(
        cls,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        context: str,
    ) -> ChemicalInfo:
        chemical_info = ChemicalInfo()
        repetitions_list: List[int] = []
        for schema in schemas:
            new_chemical: Chemical = Chemical()
            dropwise = new_chemical.get_chemical(schema, schema_parser)
            print(new_chemical.name)
            if len(schemas) > 1:
                repetitions = new_chemical.get_quantity(schema, amount_parser)
            else:
                repetitions = new_chemical.get_quantity(context, amount_parser)
            if new_chemical.name == "":
                pass
            elif new_chemical.name.strip().lower() == "n/a":
                pass
            else:
                chemical_info.chemical_list.append(new_chemical)
                chemical_info.dropwise.append(dropwise)
                repetitions_list.append(repetitions)
        if len(repetitions_list) == 0:
            chemical_info.repetitions = 1
        else:
            chemical_info.repetitions = max(repetitions_list)
        return chemical_info

    def validate_conditions(self, conditions_parser: ParametersParser, complex_conditions_parser: Optional[ComplexParametersParser]=None, add_others: bool=False) -> None:
        conditions: Dict[str, Any] = conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        if complex_conditions_parser is not None:
            complex_conditions = complex_conditions_parser.get_parameters(
            self.action_context
        ).__dict__
        for atribute in self.__dict__.keys():
            try:
                new_value = conditions[atribute][0]
            except Exception:
                if len(conditions["other"]) > 0 and add_others is True:
                    new_value = conditions["other"][0]
                else:
                    new_value = self.__dict__[atribute]
            if complex_conditions_parser is not None:
                try:
                    new_value = complex_conditions[atribute][0]
                except Exception:
                    pass
            setattr(self, atribute, new_value)

    
    @classmethod
    def validate_chemicals_materials(
        cls,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        context: str,
    ) -> ChemicalInfo:
        chemical_info = ChemicalInfoMaterials()
        repetitions_list: List[int] = []
        for schema in schemas:
            new_chemical: ChemicalsMaterials = ChemicalsMaterials()
            dropwise = new_chemical.get_chemical_materials(schema, schema_parser)
            if len(schemas) > 1:
                repetitions = new_chemical.get_quantity(schema, amount_parser)
            else:
                repetitions = new_chemical.get_quantity(context, amount_parser)
            if new_chemical.name == "":
                pass
            elif new_chemical.name.strip().lower() == "n/a":
                pass
            elif new_chemical._chemical_type == "final solution":
                chemical_info.final_solution = new_chemical
            else:
                chemical_info.chemical_list.append(new_chemical)
                chemical_info.dropwise.append(dropwise)
                repetitions_list.append(repetitions)
        if len(repetitions_list) == 0:
            chemical_info.repetitions = 1
        else:
            chemical_info.repetitions = max(repetitions_list)
        return chemical_info

class Treatment(ActionsWithChemicalAndConditions):
    solutions: List[ChemicalsMaterials] = []
    suspension_concentration: Optional[str] = None
    temperature: Optional[str] = None
    duration: Optional[str] = None
    repetitions: int = 1
    
    @classmethod
    def generate_treatment(
        cls,
        name,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action: Treatment = cls(action_name=name, action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info: ChemicalInfoMaterials = action.validate_chemicals_materials(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        else:
            action.solutions = chemicals_info.chemical_list
            action.repetitions = chemicals_info.repetitions
        concentration = re.findall(r'\d+', str(action.suspension_concentration))
        list_of_actions: List[Any] = []
        if len(action.solutions) > 0:
            list_of_actions.append(NewSolution(action_name="NewSolution").zeolite_dict())
            for solution in action.solutions:
                if len(concentration) > 0 and len(solution["quantity"]) == 0:
                    solution["quantity"].append(str(float(concentration[0]) / len(action.solutions)) + "mL")
                new_action: Actions = Add(action_name="Add", material=solution)
                list_of_actions.append(new_action.zeolite_dict())
        if action.temperature is not None:
            new_action = ChangeTemperature(action_name="ChangeTemperature", temperature=action.temperature)
            list_of_actions.append(new_action.zeolite_dict())
        if len(concentration) > 0:
            new_sample: Dict[str, Any] = {'action': 'Add', 'content': {'material': {'name': 'sample', 'quantity': ['1 g'], 'concentration': []}}, 'dropwise': False, 'duration': None, 'ph': None}
            list_of_actions.append(new_sample)
        if action.duration is not None:
            new_action = Stir(action_name="Stir", duration=action.duration)
            list_of_actions.append(new_action.zeolite_dict())
        return list_of_actions

### Actions for Organic Synthesis

class PH(ActionsWithChemicalAndConditions):
    ph: Optional[str] = None
    material: Optional[Chemical] = None
    dropwise: bool = False
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="PH", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0]
        else:
            action.material = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Material have been found on Partition object, only the first one was considered"
            )
        dimensionless_values = DimensionlessParser.get_dimensionless_numbers(context)
        if len(dimensionless_values) == 0:
            pass
        elif len(dimensionless_values) == 1:
            action.ph = dimensionless_values[0]
        else:
            action.ph = dimensionless_values[0]
            print(
                "Warning: More than one dimentionless value was found for the pH, only the first one was considered"
            )
        return [action.transform_into_pistachio()]


class Add(ActionsWithChemicalAndConditions):
    material: Optional[Chemical] = None
    dropwise: bool = False
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        if len(ph_parser.find_keywords(context)) > 0:
            return PH.generate_action(
                context, schemas, schema_parser, amount_parser, conditions_parser
            )
        action = cls(action_name="Add", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info: ChemicalInfo = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        list_of_actions = []
        if len(chemicals_info.chemical_list) == 0:
            list_of_actions.append(action.transform_into_pistachio())
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0]
            action.dropwise = chemicals_info.dropwise[0]
            list_of_actions.append(action.transform_into_pistachio())
        else:
            i = 0
            for chemical in chemicals_info.chemical_list:
                action.material = chemical
                action.dropwise = chemicals_info.dropwise[i]
                list_of_actions.append(action.transform_into_pistachio())
                i += 1
        return list_of_actions


class CollectLayer(Actions):
    layer: Optional[str] = None

    @validator("layer")
    def layer_options(cls, layer):
        valid_layers = ["aqueous", "organic", None]
        if layer not in valid_layers:
            raise ValueError('layer must be equal to "aqueous" or "organic"')
        return layer

    @classmethod
    def generate_action(
        cls,
        context: str,
        parser_aqueous: KeywordSearching,
        parser_organic: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="CollectLayer", action_context=context)
        aqueous_keywords = parser_aqueous.find_keywords(action.action_context)
        organic_keywords = parser_organic.find_keywords(action.action_context)
        if len(aqueous_keywords) > 0:
            action.layer = "aqueous"
        elif len(organic_keywords) > 0:
            action.layer = "organic"
        else:
            return []
        return [action.transform_into_pistachio()]


class Concentrate(Actions):
    @classmethod
    def generate_action(cls, context: str) -> List[Dict[str, Any]]:
        return [
            cls(
                action_name="Concentrate", action_context=context
            ).transform_into_pistachio()
        ]


class Degas(ActionsWithConditons):
    atmosphere: Optional[str] = None
    duration: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Degas", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.transform_into_pistachio()]


class DrySolid(ActionsWithConditons):
    """Dry a solid under air or vacuum.
    For drying under vacuum, the atmosphere variable should contain the string 'vacuum'.
    For drying on air, the atmosphere variable should contain the string 'air'.
    For other atmospheres, the corresponding gas name should be given ('N2', 'argon', etc.).
    """

    duration: Optional[str] = None
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="DrySolid", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.transform_into_pistachio()]


class DrySolution(ActionsWithchemicals):
    """Dry an organic solution with a desiccant"""

    material: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="DrySolution", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0].name
        else:
            action.material = chemicals_info.chemical_list[0].name
            print(
                "Warning: More than one Material found on DrySolution object, only the first one was considered"
            )
        return [action.transform_into_pistachio()]


class Extract(ActionsWithchemicals):
    solvent: Optional[Chemical] = None
    repetitions: int = 1

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Extract", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, context=action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.solvent = chemicals_info.chemical_list[0]
            action.repetitions = chemicals_info.repetitions
        else:
            action.solvent = chemicals_info.chemical_list[0]
            action.repetitions = chemicals_info.repetitions
            print(
                "Warning: More than one Solvent found on DrySolution object, only the first one was considered"
            )
        return [action.transform_into_pistachio()]


class Filter(Actions):
    """
    Filtration action, possibly with information about what phase to keep ('filtrate' or 'precipitate')
    """

    phase_to_keep: Optional[str] = None

    @validator("phase_to_keep")
    def phase_options(cls, phase_to_keep):
        if phase_to_keep is not None and phase_to_keep not in [
            "filtrate",
            "precipitate",
            None,
        ]:
            raise ValueError(
                'phase_to_keep must be equal to "filtrate" or "precipitate"'
            )
        return phase_to_keep

    @classmethod
    def generate_action(
        cls,
        context: str,
        filtrate_parser: KeywordSearching,
        precipitate_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Filter", action_context=context)
        filtrate_results = filtrate_parser.find_keywords(action.action_context)
        precipitate_results = precipitate_parser.find_keywords(action.action_context)
        if len(filtrate_results) > 0:
            action.phase_to_keep = "filtrate"
        elif len(precipitate_results) > 0:
            action.phase_to_keep = "precipitate"
        return [action.transform_into_pistachio()]


class MakeSolution(ActionsWithChemicalAndConditions):
    """
    Action to make a solution out of a list of compounds.
    This action is usually followed by another action using it (Add, Quench, etc.).
    """

    materials: List[Chemical] = []
    dropwise: bool = False
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None

    @validator("materials")
    def amount_materials(cls, materials):
        if len(materials) < 2:
            raise ValueError(
                f"MakeSolution requires at least two components (actual: {len(materials)}"
            )
        return materials
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="MakeSolution", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            return Add.generate_action(
                context,
                schemas,
                schema_parser,
                amount_parser,
                conditions_parser,
                ph_parser,
            )
        else:
            action.materials = chemicals_info.chemical_list
            for test in chemicals_info.dropwise:
                if test is True:
                    action.dropwise = True
                    break
        return [action.transform_into_pistachio()]


class Microwave(ActionsWithConditons):
    duration: Optional[str] = None
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Microwave", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.transform_into_pistachio()]


class Partition(ActionsWithchemicals):
    material_1: Optional[Chemical] = None
    material_2: Optional[Chemical] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Partition", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material_1 = chemicals_info.chemical_list[0]
        else:
            action.material_1 = chemicals_info.chemical_list[0]
            action.material_2 = chemicals_info.chemical_list[1]
        if len(chemicals_info.chemical_list) > 2:
            print(
                "Warning: More than two Materials have been found on Partition object, only the first two were considered"
            )
        return [action.transform_into_pistachio()]


class PhaseSeparation(Actions):
    @classmethod
    def generate_action(cls, context: str) -> List[Dict[str, Any]]:
        return [
            cls(
                action_name="PhaseSeparation", action_context=context
            ).transform_into_pistachio()
        ]


class Purify(Actions):
    @classmethod
    def generate_action(cls, context: str) -> List[Dict[str, Any]]:
        return [
            cls(action_name="Purify", action_context=context).transform_into_pistachio()
        ]


class Quench(ActionsWithChemicalAndConditions):
    material: Optional[Chemical] = None
    dropwise: bool = False
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Quench", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0]
            action.dropwise = chemicals_info.dropwise[0]
        else:
            action.material = chemicals_info.chemical_list[0]
            action.dropwise = chemicals_info.dropwise[0]
            print(
                "Warning: More than one Material found on Quench object, only the first one was considered"
            )
        print(ph_parser.find_keywords(context))
        if len(ph_parser.find_keywords(context)) > 0:
            new_action: List[Dict[str, Any]] = PH.generate_action(
                context, schemas, schema_parser, amount_parser, conditions_parser
            )
            return [action.transform_into_pistachio()] + new_action
        return [action.transform_into_pistachio()]


class Recrystallize(ActionsWithChemicalAndConditions):
    solvent: Optional[Chemical] = None
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Recrystallize", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.solvent = chemicals_info.chemical_list[0]
        else:
            action.solvent = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Solvent found on Recrystallize object, only the first one was considered"
            )
        return [action.transform_into_pistachio()]


class Reflux(ActionsWithConditons):
    duration: Optional[str] = None
    dean_stark: bool = False
    atmosphere: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Reflux", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.transform_into_pistachio()]


class Stir(ActionsWithConditons):
    duration: Optional[str] = None
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None
    pressure: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Stir", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.transform_into_pistachio()]


class SetTemperature(ActionsWithConditons):
    """
    If there is a duration given with cooling/heating, use "Stir" instead
    """

    temperature: Optional[str] = None
    duration: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        conditions_parser: ParametersParser,
        microwave_parser: KeywordSearching,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="SetTemperature", action_context=context)
        action.validate_conditions(conditions_parser)
        if action.temperature is None:
            return []
        elif len(microwave_parser.find_keywords(context)) > 0:
            return Microwave.generate_action(context, conditions_parser)
        elif action.duration is not None:
            return Stir.generate_action(context, conditions_parser)
        return [action.transform_into_pistachio()]


class Sonicate(ActionsWithConditons):
    duration: Optional[str] = None
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Sonicate", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.transform_into_pistachio()]


class Triturate(ActionsWithchemicals):
    solvent: Optional[Chemical] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Triturate", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.solvent = chemicals_info.chemical_list[0]
        else:
            action.solvent = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Solvent found on Triturate object, only the first one was considered"
            )
        return [action.transform_into_pistachio()]


class Wait(ActionsWithConditons):
    """
    NB: "Wait" as an action can be ambiguous depending on the context.
    It seldom means "waiting without doing anything", but is often "continue what was before", at least in Pistachio.
    """

    duration: Optional[str] = None
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Wait", action_context=context)
        action.validate_conditions(conditions_parser)
        if action.duration is None:
            return []
        return [action.transform_into_pistachio()]


class Wash(ActionsWithchemicals):
    material: Optional[Chemical] = None
    repetitions: int = 1

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Wash", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(schemas) == 1:
            action.material = chemicals_info.chemical_list[0]
            action.repetitions = chemicals_info.repetitions
        else:
            action.material = chemicals_info.chemical_list[0]
            action.repetitions = chemicals_info.repetitions
            print(
                "Warning: More than one Material found on Wash object, only the first one was considered"
            )
        return [action.transform_into_pistachio()]


class Yield(ActionsWithchemicals):
    material: Optional[Chemical] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="Yield", action_context=context)
        chemicals_info = action.validate_chemicals(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            action.material = chemicals_info.chemical_list[0]
        else:
            action.material = chemicals_info.chemical_list[0]
            print(
                "Warning: More than one Material found on Yield object, only the first one was considered"
            )
        return [action.transform_into_pistachio()]

### Actions for Heterogeneous Catalysts

class AddMaterials(ActionsWithChemicalAndConditions):
    material: Optional[ChemicalsMaterials] = None
    dropwise: bool = False
    atmosphere: Optional[str] = None
    temperature: Optional[str] = None
    duration: Optional[str] = None
    ph: Optional[str] = None
    
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: AddMaterials = cls(action_name="Add", action_context=context)
        action.validate_conditions(conditions_parser)
        chemicals_info: ChemicalInfoMaterials = action.validate_chemicals_materials(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if len(ph_parser.find_keywords(context)) > 0:
            dimensionless_values = DimensionlessParser.get_dimensionless_numbers(context)
            if len(dimensionless_values) == 0:
                pass
            elif len(dimensionless_values) == 1:
                action.ph = dimensionless_values[0]
            else:
                action.ph = dimensionless_values[0]
                print(
                    "Warning: More than one dimentionless value was found for the pH, only the first one was considered"
                )
        list_of_actions: List[Dict[str, Any]] = []
        if action.temperature is not None:
            list_of_actions.append(ChangeTemperature(action_name="ChangeTemperature", temperature=action.temperature).zeolite_dict())
        if action.atmosphere is not None:
            list_of_actions.append(SetAtmosphere(action_name="SetAtmosphere", atmosphere= action.atmosphere).zeolite_dict())
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(chemicals_info.chemical_list) == 1:
            banned_names: List[str] = banned_parser.find_keywords(chemicals_info.chemical_list[0].name)
            if len(banned_names) == 0:
                action.material = chemicals_info.chemical_list[0]
                action.dropwise = chemicals_info.dropwise[0]
                list_of_actions.append(action.zeolite_dict())
        else:
            i = 0
            for chemical in chemicals_info.chemical_list:
                banned_names: List[str] = banned_parser.find_keywords(chemical.name)
                if len(banned_names) == 0:
                    action.material = chemical
                    action.dropwise = chemicals_info.dropwise[i]
                    list_of_actions.append(action.zeolite_dict())
                i += 1
        return list_of_actions


class NewSolution(ActionsWithChemicalAndConditions):
    solution: Optional[ChemicalsMaterials] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
        ph_parser: KeywordSearching,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: NewSolution = cls(action_name="NewSolution", action_context=context)
        chemicals_info: ChemicalInfoMaterials = action.validate_chemicals_materials(
            schemas, schema_parser, amount_parser, action.action_context
        )
        if chemicals_info.final_solution is not None:
            action.solution = chemicals_info.final_solution
        list_of_actions: List[Dict[str, Any]] = []
        list_of_actions.append(action.zeolite_dict())
        add_actions = AddMaterials.generate_action(
                context, schemas, schema_parser, amount_parser, conditions_parser, ph_parser, banned_parser
            )
        list_of_actions += add_actions
        return list_of_actions

class Crystallization(ActionsWithConditons):
    temperature: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None
    stirring_speed: Optional[str] = None
    microwave: bool = False

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser, microwave_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: Crystallization = cls(action_name="Crystallization", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        keywords_list = microwave_parser.find_keywords(context)
        if len(keywords_list) > 0:
            action.microwave = True
        return [action.zeolite_dict()]

class Separate(Actions):
    phase_to_keep: str = "precipitate"
    method: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        filtrate_parser: KeywordSearching,
        precipitate_parser: KeywordSearching,
        centrifuge_parser: KeywordSearching,
        filter_parser: KeywordSearching,
        evaporation_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: Separate = cls(action_name="Separate", action_context=context)
        filtrate_results: List[str] = filtrate_parser.find_keywords(action.action_context)
        precipitate_results: List[str] = precipitate_parser.find_keywords(action.action_context)
        centrifuge_results: List[str] = centrifuge_parser.find_keywords(action.action_context)
        filter_results: List[str] = filter_parser.find_keywords(action.action_context)
        evaporation_results: List[str] = evaporation_parser.find_keywords(action.action_context)
        if len(filtrate_results) > 0:
            action.phase_to_keep = "filtrate"
        elif len(precipitate_results) > 0:
            action.phase_to_keep = "precipitate"
        if len(filter_results) > 0:
            action.method = "filtration"
        elif len(centrifuge_results) > 0:
            action.method = "centrifugation"
        elif len(evaporation_results) > 0:
            action.method = "evaporation"
        return [action.transform_into_pistachio()]
        

class WashMaterial(ActionsWithchemicals):
    material: Optional[ChemicalsMaterials] = None
    method: Optional[str] = None

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        centrifuge_parser: KeywordSearching,
        filter_parser: KeywordSearching,
        banned_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action: WashMaterial = cls(action_name="Wash", action_context=context)
        chemicals_info: ChemicalInfoMaterials = action.validate_chemicals_materials(
            schemas, schema_parser, amount_parser, action.action_context
        )
        centrifuge_results: List[str] = centrifuge_parser.find_keywords(action.action_context)
        filter_results: List[str] = filter_parser.find_keywords(action.action_context)
        list_of_actions: List[Any] = []
        if len(filter_results) > 0:
            action.method = "filtration"
        elif len(centrifuge_results) > 0:
            action.method = "centrifugation"
        if len(chemicals_info.chemical_list) == 0:
            pass
        elif len(schemas) == 1:
            banned_names: List[str] = banned_parser.find_keywords(chemicals_info.chemical_list[0].name)
            if len(banned_names) == 0:
                action.material = chemicals_info.chemical_list[0]
                list_of_actions.append(action.zeolite_dict())
        else:
            for material in chemicals_info.chemical_list:
                banned_names: List[str] = banned_parser.find_keywords(material.name)
                if len(banned_names) == 0:
                    action.material = material
                    list_of_actions.append(action.zeolite_dict())
        if chemicals_info.repetitions > 1:
            list_of_actions.append(Repeat(action_name="Repeat", amount=chemicals_info.repetitions).zeolite_dict())
        return list_of_actions

class WaitMaterial(ActionsWithConditons):
    duration: Optional[str] = None
    temperature: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action: WaitMaterial = cls(action_name="Wait", action_context=context)
        action.validate_conditions(conditions_parser)
        list_of_actions: List[Any] = []
        if action.temperature is not None:
            list_of_actions.append(ChangeTemperature(action_name="ChangeTemperature", temperature=action.temperature).zeolite_dict())
        if action.duration is not None:
            list_of_actions.append(action.zeolite_dict())
        return list_of_actions

class DryMaterial(ActionsWithConditons):
    temperature: Optional[str] = None
    duration: Optional[str] = None
    atmosphere: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser
    ) -> List[Dict[str, Any]]:
        action: DryMaterial = cls(action_name="Dry", action_context=context)
        action.validate_conditions(conditions_parser)
        return [action.zeolite_dict()]

class ThermalTreatment(ActionsWithConditons):
    temperature: Optional[str] = None
    duration: Optional[str] = None
    heat_ramp: Optional[str] = None
    atmosphere: Optional[List[str]] = None
    flow_rate: Optional[str] = None
    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser
    ) -> List[Dict[str, Any]]:
        action: ThermalTreatment = cls(action_name="ThermalTreatment", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        return [action.zeolite_dict()]

class StirMaterial(ActionsWithChemicalAndConditions):
    duration: Optional[str] = None
    stirring_speed: Optional[str] = None
    temperature: Optional[str] = None
    
    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser
    ) -> List[Dict[str, Any]]:
        action: StirMaterial = cls(action_name="Stir", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        list_of_actions: List[Any] = []
        if action.temperature is not None:
            list_of_actions.append(ChangeTemperature(action_name="ChangeTemperature", temperature=action.temperature).zeolite_dict())
        if action.duration is not None:
            list_of_actions.append(action.zeolite_dict())
        return list_of_actions

class IonExchange(Treatment):
    
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        return Treatment.generate_treatment("IonExchange", context, schemas, schema_parser, amount_parser, conditions_parser)
    
class AlkalineTreatment(Treatment):
    
    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        return Treatment.generate_treatment("AlkalineTreatment", context, schemas, schema_parser, amount_parser, conditions_parser)
    
class AcidTreatment(Treatment):

    @classmethod
    def generate_action(
        cls,
        context: str,
        schemas: List[str],
        schema_parser: SchemaParser,
        amount_parser: ParametersParser,
        conditions_parser: ParametersParser,
    ) -> List[Dict[str, Any]]:
        return Treatment.generate_treatment("AcidTreatment", context, schemas, schema_parser, amount_parser, conditions_parser)

class Repeat(Actions):
    amount: str = 1
    
    @classmethod
    def generate_action(cls, context: str):
        action: Repeat = cls(action_name="Repeat", action_context=context)
        number_list: List[str] = DimensionlessParser.get_dimensionless_numbers(context)
        if len(number_list) == 0:
            pass
        elif len(number_list) == 1:
            action.amount = int(float(number_list[0]))
        else:
            action.amount = int(float(number_list[0]))
            print(
                "Warning: More than one adimensional number was found, only the first one was considered"
                )
        list_of_actions: List[Any] = []
        if action.amount > 1:
            list_of_actions.append(action.zeolite_dict())
        return list_of_actions


class ChangeTemperature(ActionsWithConditons):
    temperature: Optional[str] = None
    microwave: bool = False
    heat_ramp: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None
    stirring_speed: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser, microwave_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="ChangeTemperature", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        keywords_list = microwave_parser.find_keywords(context)
        if len(keywords_list) > 0:
            action.microwave = True
        if action.duration is not None:
            new_action = Crystallization(action_name="Crystallization", temperature=action.temperature, duration=action.duration, pressure=action.pressure, stirring_speed=action.stirring_speed, microwave=action.microwave)
            return [new_action.zeolite_dict()]
        else:
            return [action.zeolite_dict()]
        
class Cool(ActionsWithConditons):
    temperature: Optional[str] = None
    microwave: bool = False
    heat_ramp: Optional[str] = None
    duration: Optional[str] = None
    pressure: Optional[str] = None
    stirring_speed: Optional[str] = None

    @classmethod
    def generate_action(
        cls, context: str, conditions_parser: ParametersParser, complex_conditions_parser: ComplexParametersParser, microwave_parser: KeywordSearching
    ) -> List[Dict[str, Any]]:
        action = cls(action_name="ChangeTemperature", action_context=context)
        action.validate_conditions(conditions_parser, complex_conditions_parser=complex_conditions_parser)
        keywords_list = microwave_parser.find_keywords(context)
        if len(keywords_list) > 0:
            action.microwave = True
        if action.duration is not None:
            new_action = Crystallization(action_name="Crystallization", temperature=action.temperature, duration=action.duration, pressure=action.pressure, stirring_speed=action.stirring_speed, microwave=action.microwave)
            return [new_action.zeolite_dict()]
        else:
            if action.temperature is None:
                action.temperature = "Cool"
            return [action.zeolite_dict()]

class SetAtmosphere(Actions):
    atmosphere: Optional[str] = None
    pressure: Optional[str] = None
    flow_rate: Optional[str] = None

class MicrowaveMaterial(ActionsWithConditons):
    pass

class Grind(Actions):

    @classmethod
    def generate_action(cls, context: str):
        action: Grind = cls(action_name="Grind", action_context=context)
        return [action.zeolite_dict()]

class Sieve(ActionsWithConditons):
    size: Optional[str] = None
    @classmethod
    def generate_action(cls, context: str, conditions_parser: ParametersParser):
        action = cls(action_name="Sieve", action_context=context)
        action.validate_conditions(conditions_parser, add_others=True)
        return [action.zeolite_dict()]

BANNED_CHEMICALS_REGISTRY: List[str] = [
    "newsolution",
    "new solution",
    "heated",
    "cooled",
    "hydrothermal",
    "gel",
    "unknown",
    "rinse",
    "teflon",
    "teflon-lined",
    "autoclave",
    "washing",
    "mixed",
    "precursor",
    "pre-prepared",
    "prepapared"
]

ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "cool": SetTemperature,
    "heat": SetTemperature,
    "settemperature": SetTemperature,
    "stir": Stir,
    "concentrate": Concentrate,
    "evaporate": Concentrate,
    "drysolution": DrySolution,
    "dry": DrySolution,
    "collectlayer": CollectLayer,
    "collect": CollectLayer,
    "extract": Extract,
    "wash": Wash,
    "makesolution": MakeSolution,
    "filter": Filter,
    "recrystallize": Recrystallize,
    "crystallize": Recrystallize,
    "recrystalize": Recrystallize,
    "purify": Purify,
    "quench": Quench,
    "phaseseparation": PhaseSeparation,
    "adjustph": PH,
    "reflux": Reflux,
    "drysolid": DrySolid,
    "degas": Degas,
    "partition": Partition,
    "sonicate": Sonicate,
    "triturate": Triturate,
    "wait": Wait,
    "finalproduct": Yield,
}
PISTACHIO_ACTION_REGISTRY: Dict[str, Any] = {
    "add": Add,
    "pour": Add,
    "dissolve": Add,
    "dilute": Add,
    "cool": SetTemperature,
    "heat": SetTemperature,
    "settemperature": SetTemperature,
    "warm": SetTemperature,
    "stir": Stir,
    "concentrate": Concentrate,
    "evaporate": Concentrate,
    "remove": Concentrate,
    "drysolution": DrySolution,
    "dry": DrySolution,
    "solidify": DrySolution,
    "collectlayer": CollectLayer,
    "collect": CollectLayer,
    "extract": Extract,
    "wash": Wash,
    "filter": Filter,
    "recrystallize": Recrystallize,
    "crystallize": Recrystallize,
    "recrystalize": Recrystallize,
    "purify": Purify,
    "distill": Purify,
    "quench": Quench,
    "phaseseparation": PhaseSeparation,
    "adjustph": PH,
    "reflux": Reflux,
    "drysolid": DrySolid,
    "degas": Degas,
    "partition": Partition,
    "sonicate": Sonicate,
    "wait": Wait,
    "finalproduct": Yield,
    "finalproduct:": Yield,
    "provide": Yield,
    "afford": Yield,
    "obtain": Yield,
}
MATERIAL_ACTION_REGISTRY: Dict[str, Any] = {
    "add": AddMaterials,
    "newsolution": NewSolution,
    "makesolution": NewSolution,
    "crystallization": Crystallization,
    "separate": Separate,
    "wash": WashMaterial,
    "wait": WaitMaterial,
    "dry": DryMaterial,
    "calcination": ThermalTreatment,
    "stir": StirMaterial,
    "ionexchange": IonExchange,
    "alkalinetreatment": AlkalineTreatment,
    "acidtreatment": AcidTreatment,
    "repeat": Repeat,
    "cool": Cool,
    "heat": ChangeTemperature,
    "grind": Grind,
    "sieve": Sieve

}
AQUEOUS_REGISTRY: List[str] = ["aqueous", "aq", "hydrophilic", "water", "aquatic"]
ORGANIC_REGISTRY: List[str] = ["organic", "org", "hydrophobic"]
FILTRATE_REGISTRY: List[str] = [
    "filtrate",
    "lixiviate",
    "percolate",
    "permeate",
    "liquid",
]
PRECIPITATE_REGISTRY: List[str] = [
    "precipitate",
    "residue",
    "filter cake",
    "sludge",
    "solid",
    "powder"
]
FILTER_REGISTRY: List[str] = [
    "filtrate",
    "filter",
    "filtration"
]
CENTRIFUGATION_REGISTRY: List[str] = [
    "centrifuge",
    "centrifugation",
    "centrifugally",
    "centrifugal"
]
EVAPORATION_REGISTRY: List[str] = [
    "evaporation",
    "evaporate"
]
MICROWAVE_REGISTRY: List[str] = ["microwave", "microwaves"]
PH_REGISTRY: List[str] = ["ph"]