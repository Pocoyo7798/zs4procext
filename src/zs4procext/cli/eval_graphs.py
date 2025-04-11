import json
from typing import Any, Dict
from zs4procext.parser import KeywordSearching

import click
import pandas as pd

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font

from zs4procext.evaluator_graphs import Evaluator_Graphs
import os


@click.command()
@click.argument('reference_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--label-threshold', default=0.7, help='Threshold for matching labels.')
@click.option('--series-threshold', default=0.7, help='Threshold for matching series.')
@click.option('--point-distance-threshold', default=0.1, help='Threshold for matching points based on euclidean distance.')

def main(reference_file: str, test_file: str, output_file: str, label_threshold: float, series_threshold: float, point_distance_threshold: float):
    """
    REFERENCE_FILE: Path to the reference JSON file.
    TEST_FILE: Path to the test JSON file.
    OUTPUT_FILE: Path to the output Excel file.
    """
    eval_graphs(reference_file, test_file, output_file, label_threshold, series_threshold, point_distance_threshold)

def eval_graphs(
    reference_file: str,
    test_file: str,
    output_file: str,
    label_threshold: float,
    series_threshold: float,
    point_distance_threshold: float
) -> None:
    reference_data = Evaluator_Graphs.load_json(reference_file)
    test_data = Evaluator_Graphs.load_json(test_file)
    
    evaluator = Evaluator_Graphs(reference_data=reference_data, threshold=label_threshold, distance_threshold=point_distance_threshold)
    
    reference_labels = evaluator.extract_labels(reference_data)
    test_labels = evaluator.extract_labels(test_data)

    reference_series = evaluator.extract_series(reference_data)
    test_series = evaluator.extract_series(test_data)
    
    tp_l, fp_l, fn_l, matches_labels, matched_refs_labels, matched_tests_labels = evaluator.match_references_tests(reference_labels, test_labels)
    tp_s, fp_s, fn_s, matches_series, matched_refs_series, matched_tests_series = evaluator.match_references_tests(reference_series, test_series)
    
    overall_tp, overall_fp, overall_fn = evaluator.point_matching_accuracy(test_data, matches_series, matched_refs_series, matched_tests_series)
    
    metrics_labels = evaluator.evaluate(tp_l, fp_l, fn_l)
    label_metrics = {
        "new_detections": metrics_labels["New detections"],
        "miss_detections": metrics_labels["Miss detections"],
        "incorrect_detections": metrics_labels["Incorrect detections"]
    }

    metrics_series = evaluator.evaluate(tp_s, fp_s, fn_s)
    series_metrics = {
        "new_detections": metrics_series["New detections"],
        "miss_detections": metrics_series["Miss detections"],
        "incorrect_detections": metrics_series["Incorrect detections"]
    }

    metrics_points = evaluator.evaluate(overall_tp, overall_fp, overall_fn)
    point_metrics = {
        "new_detections": metrics_points["New detections"],
        "miss_detections": metrics_points["Miss detections"],
        "incorrect_detections": metrics_points["Incorrect detections"]
    }

    results = {
        "label_metrics": label_metrics,
        "series_metrics": series_metrics,
        "point_metrics": point_metrics,
        "label_threshold": label_threshold,
        "series_threshold": series_threshold,
        "point_distance_threshold": point_distance_threshold,
    }

    print(results)

    # Restructure the DataFrame
    data = {
        "Metric": ["FN", "FP", "TP", "New Detections", "Miss Detections", "Incorrect Detections"],
        "Label Metrics": [fn_l, fp_l, tp_l, metrics_labels["New detections"], metrics_labels["Miss detections"], metrics_labels["Incorrect detections"]],
        "Series Metrics": [fn_s, fp_s, tp_s, metrics_series["New detections"], metrics_series["Miss detections"], metrics_series["Incorrect detections"]],
        "Point Metrics": [overall_fn, overall_fp, overall_tp, metrics_points["New detections"], metrics_points["Miss detections"], metrics_points["Incorrect detections"]],
        "Label Threshold": [label_threshold, "", "", "", "", ""],
        "Series Threshold": [series_threshold, "", "", "", "", ""],
        "Point Distance Threshold": [point_distance_threshold, "", "", "", "", ""]
    }

    df = pd.DataFrame(data)
    
    # Ensure the output file has a proper extension and filename
    if os.path.isdir(output_file):
        output_file = os.path.join(output_file, 'results.xlsx')
    elif not output_file.endswith(('.xlsx', '.xls')):
        output_file += '.xlsx'
    
    df.to_excel(output_file, index=False)

    # Apply color and bold formatting to specific rows in the Excel file
    wb = load_workbook(output_file)
    ws = wb.active
    
    bold_font = Font(bold=True)
    
    # Apply bold formatting to Precision, Recall, and F-Score rows
    for row in [4, 5, 6]:  # Rows to bold (Precision, Recall, F-Score)
        for col in range(1, len(df.columns) + 1):
            cell = ws.cell(row=row + 1, column=col)  # +1 because Excel is 1-indexed
            cell.font = bold_font
    
    # Apply gradient color to numeric cells
    for row in range(5, ws.max_row + 1):
        for col in range(2, 5):  # Columns with numeric values
            cell = ws.cell(row=row, column=col)
            if isinstance(cell.value, (int, float)):
                if cell.value >= 0.95:
                    fill = PatternFill(start_color="C3E6CB", end_color="C3E6CB", fill_type="solid")  # Pastel green
                elif cell.value >= 0.90:
                    fill = PatternFill(start_color="D4E9D6", end_color="D4E9D6", fill_type="solid")  # Lighter green
                elif cell.value >= 0.75:
                    fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")  # Light yellow
                elif cell.value >= 0.50:
                    fill = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")  # Light red
                else:
                    fill = PatternFill(start_color="F5C6CB", end_color="F5C6CB", fill_type="solid")  # Pastel red
                cell.fill = fill
    
    # Save the workbook
    wb.save(output_file)

if __name__ == "__main__":
    main()

