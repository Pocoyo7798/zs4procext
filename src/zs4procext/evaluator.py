import ast
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set
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

    def evaluate(self, tp: int, fp:int, fn:int) -> Dict[str, float]:
        if tp == 0:
            precision: float = 0
            recall: float = 0
            f_score: float = 0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f-score": f_score}

    def transform_chemical_name(self, name: str):
        name = name.lower()
        name = name.replace("(", "")
        name = name.replace(")", "")
        list_keywords: List[str] = self._keyword_parser.find_keywords(name)
        for keyword in list_keywords:
            name = name.replace(keyword, CHEMICALS_REGISTRY[keyword])
        name = "".join(dict.fromkeys(name))
        return name.replace(" ", "")
    
    def evaluate_string_list(self, test_list: List[str], ref_list: List[str], threshold: float = 0.9) -> Dict[str, int]:
        tp: int = 0
        fp: int = len(test_list)
        fn: int = len(ref_list)
        for test_string in test_list:
            test_string = test_string.replace(" ", "").lower()
            i = 0
            for ref_string in ref_list:
                ref_string = ref_string.replace(" ", "").lower()
                if SequenceMatcher(None, test_string, ref_string).ratio() > threshold:
                    tp = tp + 1
                    fp = fp - 1
                    fn = fn - 1
                    del ref_list[i]
                    break
                i += 1
        return {"true_positive": tp, "false_positive": max(0, fp), "false_negative": max(0, fn)}
    

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
                elif action["action"] in set(["ChangeTemperature", "Crystallization", "Dry", "ThermalTreatment", "SetTemperature"]):
                    ref_temp = str(ref_action["content"]["temperature"])
                    temp = str(action["content"]["temperature"])
                    if SequenceMatcher(None, temp.strip(), ref_temp.strip()).ratio() > 0.25:
                        return True, i
                else:
                    return True, i
            i = i + 1
        return False, i
    
    def exist_chemical_in_list(
        self,
        chemical: Optional[Dict[str, Any]],
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
        print(chemical)
        print(chemical_name)
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
            fn = fn + len(ref_action_list)
            fp = fp + len(action_list_transformed)
            found = 0
            for action in action_list_transformed:
                test, index = self.exist_action_in_list(
                    action, ref_action_list, threshold=threshold
                )
                if test is True:
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
                elif ref_action["action"] == "DrySolution":
                    reference_chemicals.append({'name': ref_action["content"]["material"], 'quantity': []})
                elif ref_action["action"] == "Partition":
                    reference_chemicals.append(ref_action["content"]["material_1"])
                    reference_chemicals.append(ref_action["content"]["material_2"])
            fn = fn + len(reference_chemicals)
            found = 0
            not_found = 0
            for action in action_list_transformed:
                test_2 = None
                if action["action"] in set(["Add", "Wash", "PH"]):
                    material: Dict[str, Any] = action["content"]["material"]
                    test, index = self.exist_chemical_in_list(
                        material, reference_chemicals, threshold=threshold
                    )
                elif action["action"] == "NewSolution":
                    material = action["content"]["solution"]
                    test, index = self.exist_chemical_in_list(
                        material, reference_chemicals, threshold=threshold
                    )
                elif action["action"] == "DrySolution":
                    material = {'name': action["content"]["material"], 'quantity': []}
                    test, index = self.exist_chemical_in_list(
                        material, reference_chemicals, threshold=threshold
                    )
                elif action["action"] == "Partition":
                    material_1 = action["content"]["material_1"]
                    test, index = self.exist_chemical_in_list(
                        material_1, reference_chemicals, threshold=threshold
                    )
                    if test is None:
                        pass
                    elif test is True:
                        found = found + 1
                    test = None
                    material_2 = action["content"]["material_2"]
                    test_2, index_2 = self.exist_chemical_in_list(
                        material_2, reference_chemicals, threshold=threshold
                    )
                    if test_2 is None:
                        pass
                    elif test_2 is True:
                        found = found + 1
                        del reference_chemicals[index_2]
                    test_2 = None
                else:
                    test = None
                if test is None:
                    pass
                elif test is True:
                    found = found + 1
                    del reference_chemicals[index]
                else:
                    print("############")
                    print(action["action"])
                    print(material)
                    print(reference_chemicals)
                    not_found = not_found + 1
                if test_2 is None:
                    pass
                elif test_2 is True:
                    found = found + 1
                    del reference_chemicals[index_2]
                else:
                    print("############")
                    print(action["action"])
                    print(material)
                    print(reference_chemicals)
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
    
    def evaluate_chemicals_in_ratios(self, chemicals_list: List[str], molar_ratios_list: List[Dict[str, str]], threshold: float=0.9) -> Dict[str, int]:
        if len(molar_ratios_list) == 0:
            raise AttributeError("The molar ratio list is empty, nothing to evaluate")
        i: int = 0
        i_best: int = 0
        tp_best: int = 0
        fp_best: int = 0
        fn_best: int = 0
        for molar_ratio in molar_ratios_list:
            ref_chemicals: List[str] = list(molar_ratio.keys())
            result = self.evaluate_string_list(chemicals_list, ref_chemicals, threshold=threshold)
            test = False
            if tp_best < result["true_positive"]:
                test = True
            elif tp_best > result["true_positive"]:
                pass
            elif fn_best > result["false_negative"]:
                test = True
            elif fn_best < result["false_negative"]:
                pass
            elif fp_best > result["false_positive"]:
                test = True
            if test is True:
                i_best = i
                tp_best = result["true_positive"]
                fp_best = result["false_positive"]
                fn_best = result["false_negative"]
            i += 1
        return {"true_positive": tp_best, "false_positive": fp_best, "false_negative": fn_best, "index": i_best}
    
    def evaluate_ratio(self, test_ratio: Dict[str, Any], ref_ratio: Dict[str, Any], threshold: float=0.9):
        test_keys: List[str] = list(test_ratio.keys())
        ref_keys: List[str] = list(ref_ratio.keys())
        tp: int = 0
        fp: int = len(test_keys)
        fn: int = len(ref_keys)
        for test_key in test_keys:
            test_value: str = str(test_ratio[test_key])
            i = 0
            ref_value: Optional[str] = None 
            for ref_key in ref_keys:
                if test_key == ref_key:
                    ref_value = str(ref_ratio[ref_key])
                    del ref_keys[i]
                i += 1
            if ref_value is None:
                pass
            elif SequenceMatcher(None, test_value.lower(), ref_value.lower()).ratio() > threshold:
                tp += 1
                fp -= 1
                fn -= 1
        return {"true_positive": tp, "false_positive": max(0, fp), "false_negative": max(0, fn)}
    

    def evaluate_molar_ratio_list(self, test_list: List[Dict[str, str]], ref_list: List[Dict[str, str]], threshold: float=0.9):
        fp: int = len(test_list)
        fn: int = len(ref_list)
        tp_chemicals: int = 0
        fp_chemicals: int = 0
        fn_chemicals: int = 0
        tp_ratios: int = 0
        fp_ratios: int = 0
        fn_ratios: int = 0
        j = 0
        for test_ratio in test_list:
            test_chemicals: List[str] = list(test_ratio.keys())
            if len(ref_list) > 0:
                chemicals_result: Dict[str, Any] = self.evaluate_chemicals_in_ratios(test_chemicals, ref_list, threshold=threshold)
                ref_ratio =  ref_list[chemicals_result["index"]]
                ratios_result = self.evaluate_ratio(test_ratio, ref_ratio, threshold=threshold)
                del ref_list[chemicals_result["index"]]
                fn -= 1
                fp -= 1
                tp_chemicals += chemicals_result["true_positive"]
                fp_chemicals += chemicals_result["false_positive"]
                fn_chemicals += chemicals_result["false_negative"]
                tp_ratios += ratios_result["true_positive"]
                fp_ratios += ratios_result["false_positive"]
                fn_ratios += ratios_result["false_negative"]
                j += 1
        if fp > 0:
            for  molar_ratio in test_list[j:]:
                fp_chemicals += len(molar_ratio.keys())
                fp_ratios += len(molar_ratio.keys())
        if fn > 0:
            for molar_ratio in ref_list:
                fn_chemicals += len(molar_ratio.keys())
                fn_ratios += len(molar_ratio.keys())
        return {"true_positive": tp_chemicals, "false_positive": fp_chemicals, "false_negative": fn_chemicals}, {"true_positive": tp_ratios, "false_positive": fp_ratios, "false_negative": fn_ratios}

    def evaluate_molar_ratio(self, test_dataset_path: str):
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        i = 0
        tp_chemicals: int = 0
        fp_chemicals: int = 0
        fn_chemicals: int = 0
        tp_ratios: int = 0
        fp_ratios: int = 0
        fn_ratios: int = 0
        tp_equations: int = 0
        fp_equations: int = 0
        fn_equations: int = 0
        for molar_dict in test_dataset:
            ref_molar_dict: Dict[str, Any] = ast.literal_eval(
                reference_dataset[i]
            )
            test_molar_dict: Dict[str, Any] = ast.literal_eval(
                molar_dict
            )
            molar_ratio_test: List[Dict[str,str]] = test_molar_dict["molar_ratios"]
            molar_ratio_ref: List[Dict[str,str]] = ref_molar_dict["molar_ratios"]
            equations_test: List[str] = test_molar_dict["equations"]
            equations_ref: List[str] = ref_molar_dict["equations"]
            chemicals_results, ratios_results = self.evaluate_molar_ratio_list(molar_ratio_test, molar_ratio_ref, threshold=0.9)
            equations_results: Dict[str, Any] = self.evaluate_string_list(equations_test, equations_ref)
            tp_chemicals += chemicals_results["true_positive"]
            fp_chemicals += chemicals_results["false_positive"]
            fn_chemicals += chemicals_results["false_negative"]
            tp_ratios += ratios_results["true_positive"]
            fp_ratios += ratios_results["false_positive"]
            fn_ratios += ratios_results["false_negative"]
            tp_equations += equations_results["true_positive"]
            fp_equations += equations_results["false_positive"]
            fn_equations += equations_results["false_negative"]
            i += 1
        return {"chemicals" : self.evaluate(tp_chemicals, fp_chemicals, fn_chemicals), "ratios" : self.evaluate(tp_ratios, fp_ratios, fn_ratios), "equations" : self.evaluate(tp_equations, fp_equations, fn_equations)}

    def evaluate_classifier(self, test_dataset_path: str):
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        true_positive: int = 0
        false_positive: int = 0
        false_negative: int = 0
        i: int = 0
        for test in test_dataset:
            if test == reference_dataset[i]:
                true_positive += 1
            elif test == "True\n":
                print("##########")
                print(test)
                print(i + 1)
                false_positive += 1
            else:
                false_negative += 1
            i += 1
        return self.evaluate(true_positive, false_positive, false_negative)

    def evaluate_samples(self, test_dataset_path: str):
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        i = 0
        true_positive: int = 0
        false_positive: int = 0
        false_negative: int = 0
        size = len(test_dataset)
        count = 1
        for sample_list in test_dataset:
            print(f"text processed: {count}/{size}")
            ref_sample_list: List[Dict[str, Any]] = ast.literal_eval(
                reference_dataset[i]
            )
            test_sample_list: List[Dict[str, Any]] = ast.literal_eval(
                sample_list
            )
            if max(0, len(ref_sample_list) - len(test_sample_list)) > 0:
                print(max(0, len(ref_sample_list) - len(test_sample_list)))
                print("Reference Samples")
                for ref_sample in ref_sample_list:
                    print(ref_sample["procedure"])
                print("Test Samples")
                for test_sample in test_sample_list:
                    print(test_sample["procedure"])
            true_positive += abs(len(ref_sample_list) - len(test_sample_list))
            false_positive += max(0, len(test_sample_list) - len(ref_sample_list))
            false_negative += max(0, len(ref_sample_list) - len(test_sample_list))
            count += 1
            i += 1
        return self.evaluate(true_positive, false_positive, false_negative)

    def evaluate_extractor_keys(self, test_dataset_path: str):
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        tp: int = 0
        fp: int = 0
        fn: int = 0
        i : int = 0
        for data in test_dataset:
            data_dict: Dict[str, Any] = ast.literal_eval(data)
            ref_data_dict: Dict[str, Any] = ast.literal_eval(reference_dataset[i])
            data_keys: List[str] =  list(data_dict.keys())
            ref_data_keys: List[str] =  list(ref_data_dict.keys())
            evaluation_results:  Dict[str, Any]  = self.evaluate_string_list(data_keys, ref_data_keys)
            tp += evaluation_results["true_positive"]
            fp += evaluation_results["false_positive"]
            fn += evaluation_results["false_negative"]
            i += 1
        return self.evaluate(tp, fp, fn)
    
    def evaluate_extractor_data(self, test_dataset_path: str):
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        tp: int = 0
        fp: int = 0
        fn: int = 0
        i : int = 0
        for data in test_dataset:
            data_dict: Dict[str, Any] = ast.literal_eval(data)
            ref_data_dict: Dict[str, Any] = ast.literal_eval(reference_dataset[i])
            test_set: Set[str] = set(data_dict)
            ref_set = set()
    
    def verify_dict_in_list(self, dictionary: Dict[str, Any], dictionary_list: List[Dict[str, Any]], threshold: float = 0.8):
        dictionary_keys = list(dictionary.keys())
        i: int = 0
        final_index: Optional[int] = None
        for entry in dictionary_list:
            test = False
            for key in dictionary_keys:
                try:
                    dictionary_value: List[str] = dictionary[key]
                    entry_value = entry[key]
                    comparison_results: Dict[str, Any] = self.evaluate_string_list(dictionary_value, entry_value, threshold= threshold)
                    if comparison_results["false_negative"] == 0 and comparison_results["false_positive"] == 0:
                        test = True
                        break
                except KeyError:
                    pass
            if test is True:
                final_index = i
                break
            i += 1
        return final_index
    
    def evaluate_dcitionary_list(self, test_dictionaries: List[Dict[str, Any]], ref_dictionaries: List[Dict[str, Any]]):
        tp_keys: int = 0
        fp_keys: int = 0
        fn_keys: int = 0
        tp_data: int = 0
        fp_data: int = 0
        fn_data: int = 0
        for dictionary in test_dictionaries:
            dictionary_keys = list(dictionary.keys())
            index = self.verify_dict_in_list(dictionary, ref_dictionaries)
            if index is None:
                fp_keys += len(dictionary_keys)
                for key in dictionary_keys:
                    fp_data += len(dictionary[key])
            else:
                ref_dictionary = ref_dictionaries[index].copy()
                del ref_dictionaries[index]

    def evaluate_table_extractor(self, test_dataset_path: str):
        with open(self.reference_dataset_path, "r") as f:
            reference_dataset: List[str] = f.readlines()
        with open(test_dataset_path, "r") as f:
            test_dataset: List[str] = f.readlines()
        tp_keys: int = 0
        fp_keys: int = 0
        fn_keys: int = 0
        tp_data: int = 0
        fp_data: int = 0
        fn_data: int = 0
        i : int = 0
        for data in test_dataset:
            data_dict: Dict[str, Any] = ast.literal_eval(data)
            ref_data_dict: Dict[str, Any] = ast.literal_eval(reference_dataset[i])
            test_results: List[Dict[str, Any]] = data_dict["data"]
            ref_results: List[Dict[str, Any]] = ref_data_dict["data"]
            self.evaluate_dict_lists(test_results, ref_results)

CHEMICALS_REGISTRY = {"solution": "",
                      "powder": "",
                      "hot": "",
                      "cyanide": "CN",
                      "saturated": "",
                      "salt": "",
                      "nanorods": "",
                      "dispersion": "",
                      "of": "",
                      "phosphoric acid": "h3po4",
                      "chloroplatinic acid": "h2ptcl6∙6h2o",
                      "sodium tetrachloropalladate": "na2pdcl4",
                      "vanadium": "v",
                      "hexachlororhodate": "rhcl6",
                      "tetra": "4",
                      "cerium": "ce",
                      "tri": "3",
                      "carbon nanotube": "cnt",
                      "crushed": "",
                      "aqueous": "",
                      "⋅": "",
                      "sample": "",
                      "dilute": "",
                      "ethyl acetate": "etoac",
                      "concentrated": "",
                      "deionized" : "",
                      "anhydrous": "",
                      "sodium": "na",
                      "dichloromethane": "dcm",
                      "borohydride": "bh4",
                      "bicarbonate": "hco3",
                      "tetrahydrofuran": "thf",
                      "cetyl trimethyl ammonium bromide" : "ctab",
                      "sodiu metasilicate": "na2sio3",
                      "cetrimonium bromide": "ctab",
                      "water": "h2o",
                      "fluoride": "f",
                      "hydroxide": "oh",
                      "sulfuric acid": "h2so4",
                      "ii": "",
                      "iii": "",
                      "iv": "",
                      "dioxide": "O2",
                      "palladium": "pd",
                      "nitric acid": "hno3",
                      "hydrochloric acid": "hcl",
                      "hydrofluoric acid": "hf",
                      "tetramethylammonium": "tma",
                      "tetrapropylammonium": "tpa",
                      "tetrabutylammonium": "tba",
                      "aluminum sulfate": "al2(so4)3",
                      "aluminum sulphate": "al2(so4)3",
                      "titanium(IV) n-butoxide": "ti(obu)4",
                      "n-butoxide": "obu",
                      "titanium": "ti",
                      "ammonium": "nh4",
                      "nitrate": "no3",
                      "bromide": "br",
                      "hydrate": "h2o",
                      "alumina": "al2o3",
                      "aluminate": "alo2",
                      "silica": "sio4",
                      "metasilicate": "sio3",
                      "penta": "5",
                      "hexa": "6",
                      "silicate": "sio3",
                      "tetraethylorthosilicate": "teos",
                      "tetraethyl": "te",
                      "orthosilicate": "os",
                      "nickel": "ni",
                      "ni(ii)": "ni",
                      "tin(ii)": "sn",
                      "tin": "sn",
                      "iron": "fe",
                      "zinc": "zn",
                      "chloride": "cl",
                      "citrate": "c12h10o14",
                      "triphenylphosphine": "pph3",
                      "triphenyl phosphine": "pph3",
                      "oxide": "o",
                      "aluminium": "al",
                      "aluminum": "al",
                      "copper": "cu",
                      "potassium": "k",
                      "hydrogen": "h2",
                      "sulfate": "so4",
                      "polytetrafluoroethylene": "ptfe",
                      "cobalt": "co",
                      "manganese": "mn",
                      "oac": "ch3co2",
                      "acetate": "ch3co2",
                      "iso-propoxide": "o-ch(ch3)2",
                      "germanium": "ge",
                      "gold": "au"
}