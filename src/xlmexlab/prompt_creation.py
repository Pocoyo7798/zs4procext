"""
Parameter extraction pipeline:
  1. build_extraction_prompt()  → creates the LLM prompt for a paragraph
  2. parse_extraction_response() → parses the LLM key-value output into structured dicts
"""

import re
from typing import Any
from pydantic import BaseModel, PrivateAttr


#https://huggingface.co/princeton-nlp/gemma-2-9b-it-SimPO

# Parameter metadata
# Defines unit hints and whether multiple values are expected per parameter.
# Add / adjust entries as your schema evolves.

PARAM_META: dict[str, dict] = {
    "size_nm": {
        "description": "Nanoparticle hydrodynamic diameter or size",
        "unit_hint": "nm",
        "multi_value": True,
    },
    "zeta_potential_mv": {
        "description": "Zeta potential (surface charge)",
        "unit_hint": "mV",
        "multi_value": True,
    },
    "pdi": {
        "description": "Polydispersity index",
        "unit_hint": "dimensionless (0–1)",
        "multi_value": True,
    },
    "encapsulation_efficiency_pct": {
        "description": "Encapsulation efficiency",
        "unit_hint": "%",
        "multi_value": True,
    },
    "ic50": {
        "description": "Half-maximal inhibitory concentration (IC50)",
        "unit_hint": "µM, nM, mg/mL, or as reported",
        "multi_value": True,
    },
    "distribution_half_life_h": {
        "description": "Distribution half-life (α phase)",
        "unit_hint": "h",
        "multi_value": True,
    },
    "circulation_half_life_h": {
        "description": "Circulation / elimination half-life",
        "unit_hint": "h",
        "multi_value": True,
    },
    "dose_group": {
        "description": "Administered dose",
        "unit_hint": "mg/kg, mg/m², µg or as reported",
        "multi_value": True,
    },
    "tumor_vol_reduction_pct": {
        "description": "Tumour volume reduction relative to control",
        "unit_hint": "%",
        "multi_value": True,
    },
    "delivery_efficiency": {
        "description": "Cellular or in-vivo delivery efficiency",
        "unit_hint": "% or fold-change",
        "multi_value": True,
    },
    "biodistribution": {
        "description": "Biodistribution/ accumulation on the different organs",
        "unit_hint": "% or % ID",
        "multi_value": True,        
    }
}

class PromptCreation(BaseModel):
    
    def build_extraction_prompt_json(self, paragraph: str, extracted_flags: dict[str, Any]) -> tuple[dict, list[str]]:
        """
        Build extraction prompt as JSON using the EXACT structure requested.
        Maintains the same content/rules as the original prompt.
        """

        targets = [k for k, v in extracted_flags.items() if v is True]

        if not targets:
            return {}

        definitions = {}

        for param in targets:
            meta = PARAM_META.get(param, {})
            desc = meta.get("description", param.replace("_", " "))
            unit = meta.get("unit_hint", "as reported")
            multi = meta.get("multi_value", False)

            multi_note = (
                "There may be multiple values reported under different conditions "
                "(e.g. different pH, temperature, time-point). List ALL of them."
                if multi
                else "Report only the primary value."
            )

            definitions[param] = {
                "Description": desc,
                "Expected unit": unit,
                "Multiplicity": multi_note
            }

        prompt_json = {
            "expertise": "You are a nanoparticles information-extraction assistant.",

            "initialization": (
                "Extract information truthfully from the provided paragraph only. "
                "Do not infer unsupported values."
            ),

            "objective": "Extract ONLY the parameters listed below from the provided paragraph.",

            "definitions": definitions,

            "answer_schema": {
                "Initialization": (
                    "Return strict key-value pairs, one entry per line."
                ),
                "Format": (
                    "<parameter_name> | <value> | <unit> | <condition>"
                )
            },

            "conclusion": (
                "Rules: Use the EXACT parameter names listed below. "
                "<value> must be numeric exactly as reported including deviation and ranges "
                "(e.g. 155 ± 0.1, -18.3, 0.12 - 0.14). "
                "<unit> must be as reported. "
                "<condition> must contain qualifying conditions "
                "(pH, temperature, time-point, cell line, sample type, formulation, etc.), or 'none' if absent. "
                "If multiple values exist, output one line per value. "
                "If mentioned but not numeric/cannot be extracted, write "
                "<parameter_name> | not_extractable | - | -. "
                "Do NOT add explanations, headers, or extra text."
            ),

            "paragraph": paragraph
        }

        return prompt_json, targets 