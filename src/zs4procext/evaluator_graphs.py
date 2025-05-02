import json
import Levenshtein
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field


class Evaluator_Graphs(BaseModel):
    reference_data: Dict[str, Any]
    threshold: float = Field(default=0.999)
    distance_threshold: float = Field(default=0.1)

    def evaluate(self, tp: int, fp: int, fn: int) -> Dict[str, float]:
        if tp == 0:
            ID: float = 0
            MD: float = 0
            ND: float = 0
        else:
            ND = fp / (tp + fp)
            MD = fn / (tp + fn)
            ID = (fn + fp) / (tp + fn + fp)
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

    def match_references_tests(
        self, references: List[str], tests: List[str]
    ) -> Tuple[int, int, int, List[Tuple[str, str, float]], Set[int], Set[int]]:
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

    def point_matching_accuracy(
        self,
        ref_data: Dict[str, Any],
        test_data: Dict[str, Any],
        matches: List[Tuple[str, str, float]],
        matched_ref_series: Set[int],
        matched_test_series: Set[int],
    ) -> Tuple[int, int, int]:
        overall_TP = 0
        overall_FP = 0
        overall_FN = 0

        for ref_series, test_series, _ in matches:
            ref_points = []
            test_points = []

            # Extract reference points
            for plot_name, plot_data in ref_data.items():
                if ref_series in plot_data:
                    ref_values = list(plot_data[ref_series].values())
                    ref_x = ref_values[0]
                    ref_y = ref_values[1]
                    ref_points = list(zip(ref_x, ref_y))
                    break

            # Extract test points
            for plot_name, plot_data in test_data.items():
                if test_series in plot_data:
                    test_values = list(plot_data[test_series].values())
                    test_x = test_values[0]
                    test_y = test_values[1]
                    test_points = list(zip(test_x, test_y))
                    break

            # Determine the maximum values for scaling
            max_x_value = max(ref_x + test_x) if ref_x + test_x else 1.0  # Avoid division by zero
            max_y_value = max(ref_y + test_y) if ref_y + test_y else 1.0  # Avoid division by zero

            # Scale the points separately for x and y
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

                if closest_test_point is not None:
                    test_points.pop(closest_test_index)
                    FN -= 1
                    TP += 1

            FP = len(test_points)
            overall_TP += TP
            overall_FP += FP
            overall_FN += FN

        # Handle unmatched series in reference
        for plot_name, plot_data in ref_data.items():
            for ref_idx, ref_series in enumerate(plot_data.keys()):
                if ref_idx not in matched_ref_series:
                    ref_values = list(plot_data[ref_series].values())
                    ref_points = list(zip(ref_values[0], ref_values[1]))
                    overall_FN += len(ref_points)

        # Handle unmatched series in test
        for plot_name, plot_data in test_data.items():
            for test_idx, test_series in enumerate(plot_data.keys()):
                if test_idx not in matched_test_series:
                    test_values = list(plot_data[test_series].values())
                    test_points = list(zip(test_values[0], test_values[1]))
                    overall_FP += len(test_points)

        return overall_TP, overall_FP, overall_FN

    def process_plots(self, test_data: Dict[str, Any]) -> Dict[str, int]:
        total_point_TP, total_point_FP, total_point_FN = 0, 0, 0
        total_label_TP, total_label_FP, total_label_FN = 0, 0, 0
        total_series_TP, total_series_FP, total_series_FN = 0, 0, 0

        skipped_images = 0
        total_images = len(self.reference_data)

        for plot_name in self.reference_data.keys():
            test_plot = test_data.get(plot_name)

            if not test_plot:  # Skip if test_data is empty or missing
                print(f"Skipping image '{plot_name}' – no data in test set.")
                skipped_images += 1
                continue

            ref_plot = self.reference_data[plot_name]

            # Match labels
            ref_labels = self.extract_labels({plot_name: ref_plot})
            test_labels = self.extract_labels({plot_name: test_plot})

            label_TP, label_FP, label_FN, label_matches, matched_ref_labels, matched_test_labels = self.match_references_tests(
                ref_labels,
                test_labels
            )

            print(f"{plot_name} - Label Matching: TP: {label_TP}, FP: {label_FP}, FN: {label_FN}")

            # Print matched label pairs with similarity ratio
            if label_matches:
                print(f"{plot_name} - Matched Labels:")
                for ref_l, test_l, ratio in label_matches:
                    print(f"  '{ref_l}' ↔ '{test_l}' (Similarity: {ratio:.3f})")

            # Unmatched Reference Labels
            unmatched_ref_labels = [label for idx, label in enumerate(ref_labels) if idx not in matched_ref_labels]
            if unmatched_ref_labels:
                print(f"{plot_name} - Unmatched Reference Labels:")
                for label in unmatched_ref_labels:
                    print(f"  '{label}'")

            # Unmatched Test Labels
            unmatched_test_labels = [label for idx, label in enumerate(test_labels) if idx not in matched_test_labels]
            if unmatched_test_labels:
                print(f"{plot_name} - Unmatched Test Labels:")
                for label in unmatched_test_labels:
                    print(f"  '{label}'")

            total_label_TP += label_TP
            total_label_FP += label_FP
            total_label_FN += label_FN


            # Match series
            ref_series = self.extract_series({plot_name: ref_plot})
            test_series = self.extract_series({plot_name: test_plot})
            series_TP, series_FP, series_FN, series_matches, matched_ref_series, matched_test_series = self.match_references_tests(
                ref_series,
                test_series
            )
            print(f"{plot_name} - Series Matching: TP: {series_TP}, FP: {series_FP}, FN: {series_FN}")
            print(f"{plot_name} - Matched Series:")
            for ref_s, test_s, ratio in series_matches:
                print(f"  '{ref_s}' ↔ '{test_s}' (Similarity: {ratio:.3f})")
            unmatched_refs = [s for i, s in enumerate(ref_series) if i not in matched_ref_series]
            unmatched_tests = [s for i, s in enumerate(test_series) if i not in matched_test_series]
            if unmatched_refs:
                print(f"{plot_name} - Unmatched Reference Series:")
                for s in unmatched_refs:
                    print(f"  '{s}'")
            if unmatched_tests:
                print(f"{plot_name} - Unmatched Test Series:")
                for s in unmatched_tests:
                    print(f"  '{s}'")

            total_series_TP += series_TP
            total_series_FP += series_FP
            total_series_FN += series_FN

            # Point matching
            point_TP, point_FP, point_FN = self.point_matching_accuracy(
                {plot_name: ref_plot},
                {plot_name: test_plot},
                series_matches,
                matched_ref_series,
                matched_test_series
            )
            print(f"{plot_name} - Point Matching: TP: {point_TP}, FP: {point_FP}, FN: {point_FN}")
            total_point_TP += point_TP
            total_point_FP += point_FP
            total_point_FN += point_FN

        # Final summary
        print(f"\nSkipped Images: {skipped_images}/{total_images} ({(skipped_images / total_images) * 100:.2f}%)")
        print(f"Total Label Matching - TP: {total_label_TP}, FP: {total_label_FP}, FN: {total_label_FN}")
        print(f"Total Series Matching - TP: {total_series_TP}, FP: {total_series_FP}, FN: {total_series_FN}")
        print(f"Total Point Matching - TP: {total_point_TP}, FP: {total_point_FP}, FN: {total_point_FN}")

        return {
            "Label_TP": total_label_TP,
            "Label_FP": total_label_FP,
            "Label_FN": total_label_FN,
            "Series_TP": total_series_TP,
            "Series_FP": total_series_FP,
            "Series_FN": total_series_FN,
            "Point_TP": total_point_TP,
            "Point_FP": total_point_FP,
            "Point_FN": total_point_FN,
            "Skipped_Images": skipped_images,
            "Total_Images": total_images,
            "Skipped_Percent": round((skipped_images / total_images) * 100, 2),
        }
