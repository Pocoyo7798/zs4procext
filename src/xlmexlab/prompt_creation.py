from typing import Any
from pydantic import BaseModel


DEFAULT_FIELDS = ["parameter_name", "value", "unit", "condition"]


# Global two-step screening applied to ALL parameters
GLOBAL_SCREENING = (
    "For each numerical candidate found in the text, apply these two steps:\n"
    "  STEP 1 — Is it introduced by 'X et al.', 'reported', 'showed', 'found', "
    "'according to', or a citation [N]? → DISCARD\n"
    "  STEP 2 — Is it from the authors' own experiment? → KEEP\n"
    "Only KEPT values are extracted."
)

GLOBAL_NULL_RULE = (
    "- If a parameter is not mentioned, has no numerical value, "
    "or all values were DISCARDED by the screening steps:\n"
    "  <parameter_name> | not_extractable"
)


PARAM_META: dict[str, dict] = {
    "size_nm": {
        "description": "Nanoparticle diameter or size explicitly measured in the study.",
        "unit_hint": "nm",
        "specific_format": "<parameter_name> | <value> | <unit> | <condition>",
        "field_rules": {
            "<value>": "Numeric size exactly as reported (can include ranges).",
            "<unit>": "Unit exactly as written in text.",
            "<condition>": "Experimental condition if stated, otherwise 'none'.",
        },
        "exclude": [
            "theoretical sizes",
            "expected sizes",
            "pore sizes",
            "filter sizes",
            "instrument limits",
        ],
    },

    "lipid_composition_ratio_units": {
        "description": "Quantification type used for lipid composition ratios.",
        "unit_hint": "dimensionless (molar ratio, weight ratio, etc.)",
        "specific_format": "<parameter_name> | <quantification_type>",
        "field_rules": {
            "<quantification_type>": "Type of ratio exactly as stated (e.g., mol%, molar ratio).",
        },
        "exclude": [
            "raw lipid names without ratios",
            "concentration units",
        ],
    },

    "zeta_potential_mv": {
        "description": "Zeta potential (surface charge).",
        "unit_hint": "mV",
        "specific_format": "<parameter_name> | <value> | <unit>",
        "field_rules": {
            "<value>": "Numeric zeta potential exactly as reported.",
            "<unit>": "Unit exactly as written.",
        },
        "exclude": [
            "predicted charge",
            "theoretical surface charge",
        ],
    },

    "pdi": {
        "description": "Polydispersity index.",
        "unit_hint": "dimensionless",
        "specific_format": "<parameter_name> | <value>",
        "field_rules": {
            "<value>": "Numeric PDI exactly as reported.",
        },
        "exclude": [
            "size distributions",
            "statistical variance",
        ],
    },

    "encapsulation_efficiency_pct": {
        "description": "Encapsulation efficiency.",
        "unit_hint": "%",
        "specific_format": "<parameter_name> | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric efficiency exactly as reported.",
            "<unit>": "Percentage or unit as written.",
            "<drug_name>": "Drug explicitly mentioned in experiment.",
        },
        "exclude": [
            "loading capacity",
            "drug concentration",
            "release percentage",
            "theoretical efficiency",
        ],
    },

    "ic50": {
        "description": "Half-maximal inhibitory concentration.",
        "unit_hint": "uM, nM, mg/mL, etc.",
        "specific_format": "<parameter_name> | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric IC50 exactly as reported.",
            "<unit>": "Unit exactly as written.",
            "<drug_name>": "Drug or condition tested.",
        },
        "exclude": [
            "EC50",
            "GI50",
            "CC50",
            "predicted values",
        ],
    },

    "distribution_half_life_h": {
        "description": "Distribution half-life (alpha phase).",
        "unit_hint": "h",
        "specific_format": "<parameter_name> | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric half-life exactly as reported.",
            "<unit>": "Unit in hours or as stated.",
            "<drug_name>": "Drug studied.",
        },
        "exclude": [
            "elimination half-life",
            "circulation half-life",
        ],
    },

    "circulation_half_life_h": {
        "description": "Circulation / elimination half-life.",
        "unit_hint": "h",
        "specific_format": "<parameter_name> | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric half-life exactly as reported.",
            "<unit>": "Unit as written.",
            "<drug_name>": "Drug studied.",
        },
        "exclude": [
            "distribution half-life",
        ],
    },

    "dose_group": {
        "description": "Administered treatment doses from the AUTHORS' OWN experiment only.",
        "unit_hint": "as reported",
        "specific_format": "dose_group | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric only, exactly as reported. Do NOT include units or route.",
            "<unit>": "Dose unit exactly as written. Valid formats: mg/kg, μg/kg, μg/volume, mg/m², mg/animal,  μg/100 μl, etc.",
            "<drug_name>": "Drug actually administered in this study.",
        },
        "exclude": [
            "theoretical doses",
        ],
    },

    "tumor_vol_reduction_pct": {
        "description": "Tumour volume reduction vs control.",
        "unit_hint": "%",
        "specific_format": "<parameter_name> | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric reduction exactly as reported.",
            "<unit>": "Percentage.",
            "<drug_name>": "Drug or treatment used.",
        },
        "exclude": [
            "absolute tumor volume",
            "predicted inhibition",
        ],
    },

    "delivery_efficiency": {
        "description": "Cellular or in-vivo delivery efficiency.",
        "unit_hint": "% or fold-change",
        "specific_format": "<parameter_name> | <value> | <unit> | <condition>",
        "field_rules": {
            "<value>": "Numeric efficiency exactly as reported.",
            "<unit>": "Unit or fold-change exactly as written.",
            "<condition>": "Experimental condition if present.",
        },
        "exclude": [
            "qualitative statements",
        ],
    },

    "biodistribution": {
        "description": "Organ accumulation / biodistribution.",
        "unit_hint": "% ID, %ID/g, etc.",
        "specific_format": "<parameter_name> | <value> | <unit> | <organ>",
        "field_rules": {
            "<value>": "Numeric accumulation exactly as reported.",
            "<unit>": "Unit exactly as written.",
            "<organ>": "Organ explicitly mentioned.",
        },
        "exclude": [
            "qualitative targeting",
        ],
    },
}


class PromptCreation(BaseModel):

    def build_extraction_prompt_json(
        self,
        paragraph: str,
        extracted_flags: dict[str, Any],
    ) -> tuple[dict, list[str]]:

        targets = [k for k, v in extracted_flags.items() if v is True]

        if not targets:
            return {}, []

        #  EXPERTISE (system role) 
        expertise = (
            "You are a nanoparticle information-extraction assistant. "
            "You extract data truthfully from scientific text. "
            "Only extract values explicitly stated as part of the AUTHORS' OWN experiment. "
            "Do not infer, guess, or hallucinate values."
        )

        # INITIALIZATION 
        initialization = (
            "Only extract values explicitly stated as part of the AUTHORS' OWN experiment. "
            "Ignore values mentioned from other studies, literature comparisons, hypotheses, or discussions."
        )

        #  DEFINITIONS 
        definitions: dict[str, str] = {}
        for param in targets:
            meta = PARAM_META[param]
            definitions[param] = (
                f"{meta.get('description', param.replace('_', ' '))}. "
                f"Expected unit: {meta.get('unit_hint', 'as reported')}."
            )

        # OBJECTIVE 
        objective = "Extract ONLY the parameters listed above from the paragraph below."

        # Answer schema construction
        schema_lines: list[str] = []

        # Format per parameter
        override_lines = [
            f"{PARAM_META[param]['specific_format']}"
            for param in targets
            if "specific_format" in PARAM_META.get(param, {})
        ]
        if override_lines:
            schema_lines.append("Format answers:")
            schema_lines.extend(override_lines)

        schema_lines.append("Rules:")
        schema_lines.append("- One line per extracted value.")
        schema_lines.append(
            "- For each numerical candidate found in the text, apply these two steps:\n"
            "  STEP 1 \u2014 Is it introduced by 'X et al.', 'reported', 'showed', 'found', "
            "'according to', or a citation [N]? \u2192 DISCARD\n"
            "  STEP 2 \u2014 Is it from the authors' own experiment? \u2192 KEEP\n"
            "  Only KEPT values are extracted."
        )

        # Per-parameter field rules and exclusions (format/parsing only)
        for param in targets:
            meta = PARAM_META[param]
            if "field_rules" in meta:
                schema_lines.append(f"- Field rules for {param}:")
                for field, rule in meta["field_rules"].items():
                    schema_lines.append(f"  - {field}: {rule}")
            if "exclude" in meta:
                schema_lines.append(f"- Do NOT extract for {param}:")
                for ex in meta["exclude"]:
                    schema_lines.append(f"  - {ex}")

        # Global null rule
        schema_lines.append(GLOBAL_NULL_RULE)

        answer_schema: dict[str, str] = {
            "Format": "\n".join(schema_lines),
        }

        # Conclusions
        conclusion = "Return ONLY the extraction lines. No explanations, headers, or comments."

        prompt_json = {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": definitions,
            "objective": objective,
            "answer_schema": answer_schema,
            "conclusion": conclusion,
        }

        return prompt_json, targets
    
class PromptCreationSchedule(BaseModel):

    def build_extraction_prompt_json(
        self,
        drug_names: list[str],
    ) -> tuple[dict, list[str]]:
        
        expertise = (
            "You are an expert assistant for extracting dosing schedule information from scientific text."
        )
        # INITIALIZATION 
        initialization = (
            "Given this text, for each drug below, what is the dosing schedule?"
        )

        #  DEFINITIONS 
        definitions: dict[str, str] = {}

        # OBJECTIVE 
        objective = "Only answer with: single_dose, multi_dose, or unknown."

        # Answer schema construction
        schema_lines: list[str] = []

        schema_lines.append(
            "Rules:"
            "Use ONLY explicit frequency words ('every X days', 'twice', 'q.d.', 'BID', 'once', 'single dose').\n"        
            "Do NOT use timing words ('prior to', '15 min before', 'day 0', 'for X days') as evidence.\n"
            "Format: <drug_name> | <schedule>\n"
            "Drug names: " + ", ".join(drug_names)
            )

        answer_schema: dict[str, str] = {
            "Format": "\n".join(schema_lines),
        }

        # Conclusions
        conclusion = "Return ONLY the extraction lines. No explanations, headers, or comments."

        prompt_json = {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": definitions,
            "objective": objective,
            "answer_schema": answer_schema,
            "conclusion": conclusion,
        }

        return prompt_json