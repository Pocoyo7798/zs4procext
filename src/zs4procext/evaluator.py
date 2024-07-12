import ast
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
import re
from zs4procext.parser import KeywordSearching

import numpy as np
from Levenshtein import ratio
from pydantic import BaseModel


class Evaluator(BaseModel):
    reference_dataset_path: str
    _keyword_parser: Optional[KeywordSearching] = None

    def model_post_init(self, _context):
        words_list = list(CHEMICALS_REGISTRY.keys())
        self._keyword_parser = KeywordSearching(keywords_list=words_list)
        self._keyword_parser.model_post_init(False)

    def transform_chemical_name(self, name: str):
        name = name.lower()
        list_keywords: List[str] = self._keyword_parser.find_keywords(name)
        for keyword in list_keywords:
            name = name.replace(keyword, CHEMICALS_REGISTRY[keyword])
        return name.replace(" ", "")

    def exist_action_in_list(
        self,
        action: Dict[str, Any],
        list_of_actions: List[Dict[str, Any]],
        threshold=0.8,
    ):
        """evaluates if and action is present in a action list

        Args:
            action: action to be evaluated
            list_of_actions: list of actions that could contain the action
            threshhold: threshold limit to validate if two actions are similar. Defaults to 0.8.

        Returns:
            True if an action is present inside  alist of actions
        """
        i = 0
        for ref_action in list_of_actions:
            if SequenceMatcher(None, str(action), str(ref_action)).ratio() >= threshold and action["action"] == ref_action["action"]:
                if action["action"] in set(["Stir", "Wait"]):
                    if ref_action["content"]["duration"] == action["content"]["duration"]:
                        return True, i
                    ref_duration = re.findall(r'\d+', str(ref_action["content"]["duration"]))
                    duration = re.findall(r'\d+', str(action["content"]["duration"]))
                    if len(ref_duration) > 0 and len(duration) > 0:
                        if float(ref_duration[0]) == float(duration[0]):
                            return True, i
                elif action["action"] == "Add":
                    ref_chemical_name: str = self.transform_chemical_name(ref_action["content"]["material"]["name"])
                    chemical_name: str = self.transform_chemical_name(action["content"]["material"]["name"])
                    if SequenceMatcher(None, chemical_name, ref_chemical_name).ratio() > 0.5:
                        return True, i
                elif action["action"] == "Separate":
                    ref_phase: str = ref_action["content"]["phase_to_keep"]
                    phase: str = action["content"]["phase_to_keep"]
                    if ref_phase == phase:
                        return True, i
                elif action["action"] in set(["ChangeTemperature", "Crystallization", "Dry", "ThermalTreatment"]):
                    ref_temp = str(ref_action["content"]["temperature"])
                    temp = str(action["content"]["temperature"])
                    if SequenceMatcher(None, temp.strip(), ref_temp.strip()).ratio() > 0.25:
                        return True, i
                else:
                    return True, i
            i = i + 1
        if action["action"] in set(["ChangeTemperature", "Crystallization", "Dry", "ThermalTreatment"]):
            print("########")
            print(action)
            print(list_of_actions)
            print("########")
        return False, i
    
    def exist_chemical_in_list(
        self,
        chemical: Dict[str, Any],
        list_of_chemicals: List[Dict[str, Any]],
        threshold=0.8,
    ):
        if chemical is None:
            chemical_name: str = "None"
        else:
            chemical_name = str(chemical["name"])
        chemical_name = self.transform_chemical_name(chemical_name)
        i = 0
        for ref_chemical in list_of_chemicals:
            if SequenceMatcher(None, str(chemical), str(ref_chemical)).ratio() >= threshold:
                if ref_chemical is None:
                    ref_chemical_name: str = "None"
                else:
                    ref_chemical_name: str = str(ref_chemical["name"])
                ref_chemical_name = self.transform_chemical_name(ref_chemical_name)
                if SequenceMatcher(None, chemical_name, ref_chemical_name).ratio() > 0.5:
                    return True, i
            i = i + 1
        for chemical1 in list_of_chemicals:
            if chemical1 is None:
                ref_chemical_name = "none"
            else:
                ref_chemical_name = self.transform_chemical_name(chemical1["name"])
        return False, i

    def evaluate_actions(
        self, test_dataset_path: str, threshold: float = 0.8
    ) -> Dict[str, float]:
        """evaluate the actions from a dataset

        Args:
            test_dataset_path: path to the dataset obtained after the test

        Returns:
            the precision, recall and f-score of the test dataset
        """
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        tp = 0
        fp = 0
        fn = 0
        i = 0
        for action_list in test_dataset:
            ref_action_list: List[Dict[str, Any]] = ast.literal_eval(
                reference_dataset[i]
            )
            action_list_transformed: List[Dict[str, Any]] = ast.literal_eval(
                action_list
            )
            print(i)
            #print(ref_action_list)
            #print(action_list_transformed)
            fn = fn + len(ref_action_list)
            fp = fp + len(action_list_transformed)
            found = 0
            for action in action_list_transformed:
                test, index = self.exist_action_in_list(
                    action, ref_action_list, threshold=threshold
                )
                if test is True:
                    #print(action)
                    #print(ref_action_list[index])
                    found = found + 1
                    del ref_action_list[index]
            tp = tp + found
            fp = fp - found
            fn = fn - found
            i = i + 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f-score": f_score}

    def evaluate_chemicals(
        self, test_dataset_path: str, threshold: float = 0.8
    ) -> Dict[str, float]:
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        tp = 0
        fp = 0
        fn = 0
        i = 0
        reference_chemicals: List[str] = []
        for action_list in test_dataset:
            print(i)
            ref_action_list: List[Dict[str, Any]] = ast.literal_eval(
                reference_dataset[i]
            )
            action_list_transformed: List[Dict[str, Any]] = ast.literal_eval(
                action_list
            )
            reference_chemicals: List[str] = []
            for ref_action in ref_action_list:
                if ref_action["action"] in set(["Add", "Wash"]):
                    reference_chemicals.append(ref_action["content"]["material"])
                elif ref_action["action"] == "NewSolution":
                    reference_chemicals.append(ref_action["content"]["solution"])
            fn = fn + len(reference_chemicals)
            found = 0
            not_found = 0
            for action in action_list_transformed:
                if action["action"] in set(["Add", "Wash"]):
                    material: Dict[str, Any] = action["content"]["material"]
                    test, index = self.exist_chemical_in_list(
                        material, reference_chemicals, threshold=threshold
                    )
                elif action["action"] == "NewSolution":
                    material: Dict[str, Any] = action["content"]["solution"]
                    test, index = self.exist_chemical_in_list(
                        material, reference_chemicals, threshold=threshold
                    )
                else:
                    test = None
                if test is None:
                    pass
                elif test is True:
                    found = found + 1
                    del reference_chemicals[index]
                else:
                    not_found = not_found + 1
            tp = tp + found
            fp = fp + not_found
            fn = fn - found
            i += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f-score": f_score}
                

    def evaluate_actions_order(self, test_dataset_path: str) -> Dict[str, Any]:
        """evaluate the sequence of actions of a dataset

        Args:
            test_dataset_path: path to the dataset obtained after the test

        Returns:
            the accuracy of the seuqence of actions
        """
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        i = 0
        accuracy_list: List[float] = []
        actions_amount: int = 0
        actions_over_find: int = 0
        actions_lower_find: int = 0
        for action_list in test_dataset:
            ref_action_list: List[Dict[str, Any]] = ast.literal_eval(
                reference_dataset[i]
            )
            action_list_transformed: List[Dict[str, Any]] = ast.literal_eval(
                action_list
            )
            ref_action_sequence = [action["action"] for action in ref_action_list]
            action_sequence: List[str] = [
                action["action"] for action in action_list_transformed
            ]
            ref_action_sequence2 = "".join(ref_action_sequence)
            action_sequence2 = "".join(action_sequence)
            accuracy_list.append(ratio(ref_action_sequence2, action_sequence2))
            actions_amount = actions_amount + len(ref_action_sequence)
            actions_over_find = actions_over_find + max(
                0, len(action_sequence) - len(ref_action_sequence)
            )
            actions_lower_find = actions_lower_find + max(
                0, len(ref_action_sequence) - len(action_sequence)
            )
            i = i + 1
        actions_missing = actions_lower_find / actions_amount
        actions_extra = actions_over_find / (actions_amount + actions_over_find)
        return {
            "accuracy": np.average(accuracy_list),
            "%missing": actions_missing,
            "%%extra": actions_extra,
        }

CHEMICALS_REGISTRY = {"solution": "",
                      "aqueous": "",
                      "sample": "",
                      "dilute": "",
                      "concetrated": "",
                      "sodium": "na",
                      "cetrimonium bromide": "ctab",
                      "water": "h2o",
                      "hydroxide": "oh",
                      "sulfuric acid": "h2so4",
                      "nitric acid": "hno3",
                      "hydrochloric acid": "hcl",
                      "tetramethylammonium": "tma",
                      "tetrapropylammonium": "tpa",
                      "tetrabutylammonium": "tba",
                      "ammonium": "nh4",
                      "nitrate": "no3",
                      "bromide": "br",
                      "hydrate": "h2o",
                      "alumina": "al2o3",
                      "aluminate": "alo2",
                      "silica": "sio4",
                      "metasilicate": "sio3",
                      "silicate": "sio3",
                      "tetraethyl": "te",
                      "orthosilicate": "os",
                      "nickel": "ni",
                      "ni(ii)": "ni",
                      "tin(ii)": "sn",
                      "tin": "sn",
                      "chloride": "cl",
                      "citrate": "c12h10o14",
                      "triphenylphosphine": "pph3",
                      "oxide": "o",
                      "aluminium": "al",
                      "copper": "cu",
                      "potassium": "k",
                      "hydrogen": "h2",
                      "sulfate": "so4",
                      "polytetrafluoroethylene": "ptfe",
                      "cobalt": "co",
                      "manganese": "mn",
                      "acetate": "ch3co2",
                      "iso-propoxide": "o-ch(ch3)2",
}