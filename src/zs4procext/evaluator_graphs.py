import ast
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple
import re
from zs4procext.parser import KeywordSearching

import numpy as np
import Levenshtein
from pydantic import BaseModel, Field
import json

class Evaluator_Graphs(BaseModel):
    reference_data: Dict[str, Any]
    threshold: float = Field(default=0.7)
    distance_threshold: float = Field(default=0.1)
    
    def evaluate(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        if tp == 0:
            ID: float = 0
            MD: float = 0
            ND: float = 0
        else:
            ND = tp / (tp + fp)
            MD = tp / (tp + fn)
            ID = tp/(tp + fn + fp)
        return {"New detections": ND, "Miss detections": MD, "Incorrect detections": ID}
        
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def extract_labels(data: Dict[str, Any]) -> List[str]:
        labels = []
        for plot_data in data.values():
            for data_values in plot_data.values():
                for key in data_values.keys():
                    if key not in labels:
                        labels.append(key)
        return labels

    @staticmethod
    def extract_series(data: Dict[str, Any]) -> List[str]:
        series = []
        for plot_data in data.values():
            series.extend(plot_data.keys())
        return series

    def match_references_tests(self, references: List[str], tests: List[str]) -> Tuple[int, int, int, List[Tuple[str, str, float]], Set[int], Set[int]]:
        FN = len(references)
        TP = 0
        matched_refs = set()
        matched_tests = set()
        matches = []

        for ref_idx, ref in enumerate(references):
            for test_idx, test in enumerate(tests):
                if ref_idx in matched_refs or test_idx in matched_tests:
                    continue
                ratio = Levenshtein.ratio(ref, test)
                if ratio > self.threshold:
                    matched_refs.add(ref_idx)
                    matched_tests.add(test_idx)
                    matches.append((ref, test, ratio))
                    FN -= 1
                    TP += 1
                    break

        FP = len(tests) - len(matched_tests)
        return TP, FP, FN, matches, matched_refs, matched_tests

    @staticmethod
    def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def point_matching_accuracy(self, test_data: Dict[str, Any], matches: List[Tuple[str, str, float]], matched_ref_series: Set[int], matched_test_series: Set[int]) -> Tuple[int, int, int]:
        overall_TP = 0
        overall_FP = 0
        overall_FN = 0

        for ref_series, test_series, _ in matches:
            ref_points = []
            for plot_name, plot_data in self.reference_data.items():
                if ref_series in plot_data:
                    ref_values = list(plot_data[ref_series].values())
                    ref_x = ref_values[0]
                    ref_y = ref_values[1]
                    ref_points = list(zip(ref_x, ref_y))
                    break

            test_points = []
            for plot_name, plot_data in test_data.items():
                if test_series in plot_data:
                    test_values = list(plot_data[test_series].values())
                    test_x = test_values[0]
                    test_y = test_values[1]
                    test_points = list(zip(test_x, test_y))
                    break

            # Find the maximum value for x and y separately
            max_x_value = max(max(ref_x), max(test_x))
            max_y_value = max(max(ref_y), max(test_y))

            # Scale the points by dividing by the maximum values
            ref_points = [(x / max_x_value, y / max_y_value) for x, y in ref_points]
            test_points = [(x / max_x_value, y / max_y_value) for x, y in test_points]

            FN = len(ref_points)
            TP = 0
            
            while ref_points:
                ref_point = ref_points.pop(0)
                closest_distance = float('inf')
                closest_test_point = None
                closest_test_index = None
                
                for test_index, test_point in enumerate(test_points):
                    distance = self.euclidean_distance(ref_point, test_point)
                    if distance < closest_distance and distance <= self.distance_threshold:
                        closest_distance = distance
                        closest_test_point = test_point
                        closest_test_index = test_index
                
                if closest_test_point:
                    test_points.pop(closest_test_index)
                    FN -= 1
                    TP += 1

            FP = len(test_points)
            overall_TP += TP
            overall_FP += FP
            overall_FN += FN

        return overall_TP, overall_FP, overall_FN

        for plot_name, plot_data in self.reference_data.items():
            for ref_idx, ref_series in enumerate(plot_data.keys()):
                if ref_idx not in matched_ref_series:
                    ref_values = list(plot_data[ref_series].values())
                    ref_points = list(zip(ref_values[0], ref_values[1]))
                    overall_FN += len(ref_points)

        for plot_name, plot_data in test_data.items():
            for test_idx, test_series in enumerate(plot_data.keys()):
                if test_idx not in matched_test_series:
                    test_values = list(plot_data[test_series].values())
                    test_points = list(zip(test_values[0], test_values[1]))
                    overall_FP += len(test_points)

        return overall_TP, overall_FP, overall_FN
