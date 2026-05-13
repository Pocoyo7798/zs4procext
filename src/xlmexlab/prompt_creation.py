"""
Parameter extraction pipeline:
  1. build_extraction_prompt()  → creates the LLM prompt for a paragraph
  2. parse_extraction_response() → parses the LLM key-value output into structured dicts
"""

from typing import Any
from pydantic import BaseModel


PARAM_META: dict[str, dict] = {
    "size_nm": {
        "description": "Nanoparticle hydrodynamic diameter or size",
        "unit_hint": "nm",
    },
    "zeta_potential_mv": {
        "description": "Zeta potential (surface charge)",
        "unit_hint": "mV",
    },
    "pdi": {
        "description": "Polydispersity index",
        "unit_hint": "dimensionless (0–1)",
    },
    "encapsulation_efficiency_pct": {
        "description": "Encapsulation efficiency",
        "unit_hint": "%",
    },
    "ic50": {
        "description": "Half-maximal inhibitory concentration (IC50)",
        "unit_hint": "µM, nM, mg/mL, or as reported",
    },
    "distribution_half_life_h": {
        "description": "Distribution half-life (α phase)",
        "unit_hint": "h",
    },
    "circulation_half_life_h": {
        "description": "Circulation / elimination half-life",
        "unit_hint": "h",
    },
    "dose_group": {
        "description": "Administered dose",
        "unit_hint": "mg/kg, mg/m², µg or as reported",
        "specific_format": (
            "<parameter_name> | <value> | <unit> | <drug_name> | <schedule>"
        ),
    },
    "tumor_vol_reduction_pct": {
        "description": "Tumour volume reduction relative to control",
        "unit_hint": "%",
    },
    "delivery_efficiency": {
        "description": "Cellular or in-vivo delivery efficiency",
        "unit_hint": "% or fold-change",
    },
    "biodistribution": {
        "description": "Biodistribution / accumulation in organs",
        "unit_hint": "% or % ID",
    },
}


class PromptCreation(BaseModel):

    def build_extraction_prompt_json(
        self,
        paragraph: str,
        extracted_flags: dict[str, Any]
    ) -> tuple[dict, list[str]]:

        targets = [
            k for k, v in extracted_flags.items()
            if v is True
        ]

        if not targets:
            return {}, []

        definitions = {
                "Initialization": (
                "To answer the question consider the following description"
                )
        }

        for param in targets:

            meta = PARAM_META.get(param, {})

            desc = meta.get(
                "description",
                param.replace("_", " ")
            )

            unit = meta.get(
                "unit_hint",
                "as reported"
            )

            definitions[param] = (
                f"{desc}. (expected unit: {unit})"
            )

        # Generic format unless overridden
        answer_schema = {
            "Initialization": (
                "Return strict key-value pairs, one entry per line."
            ),
            "Format": (
                "<parameter_name> | <value> | <unit> | <condition>"
            )
        }

        # Add parameter-specific formats if they exist
        special_formats = {
            param: PARAM_META[param]["specific_format"]
            for param in targets
            if "specific_format" in PARAM_META.get(param, {})
        }

        if special_formats:
            answer_schema["Parameter_specific_formats"] = special_formats

        prompt_json = {

            "expertise": (
                "You are a nanoparticles information-extraction assistant."
            ),

            "initialization": (
                "Extract information truthfully from the provided paragraph only. "
                "Do not infer unsupported values."
            ),

            "objective": (
                "Extract ONLY the parameters listed below "
                "from the provided paragraph."
            ),

            "definitions":  definitions,

            "answer_schema": answer_schema,

            "conclusion": (
                "Rules: Use the EXACT parameter names listed below. "
                "There may be multiple values reported under different conditions. List ALL of them. (ex: pH, temperature, time-point, cell line, sample type, formulation, etc.), or 'none' if absent. "
                "<value> must be numeric exactly as reported including deviation and ranges (e.g. 155 ± 0.1, -18.3, 0.12 - 0.14, >100, ≤200). "
                "<unit> must be exactly as reported. "
                "If multiple values exist, output one line per value. "
                "If mentioned but not numeric/cannot be extracted, write "
                "<parameter_name> | not_extractable | - | -. "
                "Do NOT add any explanations, headers, extra text, or comments."
                "Answer format example given before."
            ),

            "paragraph": paragraph,
        }

        return prompt_json, targets