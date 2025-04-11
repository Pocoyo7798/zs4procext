import json
import re
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import os  

class DataConverter(BaseModel):
    data_string: str
    data_dict: dict = Field(default_factory=dict)
    catalyst_label: str = ""
    x_axis_label: str = ""
    y_axis_label: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self._convert_to_dict()

    def _convert_to_dict(self):
        header_pattern = re.compile(r'^[^\n]*[a-zA-Z]+;[^\n]*[a-zA-Z]+;[^\n]*[a-zA-Z]+[^\n]*$', re.MULTILINE)
        header_match = header_pattern.search(self.data_string)
        if not header_match:
            print("No valid table header found in the input string.")
            return
        
        header_start = header_match.start()
        header_line = self.data_string[header_start:].split('\n')[0]
        header = header_line.split(';')
        if len(header) != 3:
            print("Header does not have the expected format (Catalyst;x-axis label;y-axis label).")
            return

        self.catalyst_label = header[0]
        self.x_axis_label = header[1]
        self.y_axis_label = header[2]

        data_lines = self.data_string.strip().split('\n')
        for line in data_lines:
            if not line.strip():
                continue
            if line == header_line:
                continue
            if not re.match(r'^[^;]+;[^;]+;[^;]+$', line):
                continue
            parts = line.split(';')
            if len(parts) != 3:
                continue
            catalyst = parts[0]
            try:
                x_value = float(parts[1])
                y_value = float(parts[2])
            except ValueError:
                continue
            if catalyst not in self.data_dict:
                self.data_dict[catalyst] = {self.x_axis_label: [], self.y_axis_label: []}
            self.data_dict[catalyst][self.x_axis_label].append(x_value)
            self.data_dict[catalyst][self.y_axis_label].append(y_value)

    def get_data_dict(self):
        return self.data_dict

    def to_json(self):
        return json.dumps(self.data_dict, indent=4) 