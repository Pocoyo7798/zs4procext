from typing import Any
from pydantic import BaseModel


DEFAULT_FIELDS = ["parameter_name", "value", "unit", "condition"]


# Global two-step screening applied to ALL parameters
GLOBAL_SCREENING = (
    "For each numerical candidate found in the text, apply these two steps:\n"
    "  STEP 1 — Is it introduced by 'X et al.', 'reported', 'showed', 'found', "
    "'according to', or a citation [N]? → DISCARD\n"
    "  STEP 2 — Is it from the authors' own experiment?  → KEEP\n"
    "Only KEPT values are extracted."
)

GLOBAL_NULL_RULE = (
    "- If a parameter is not mentioned, has no numerical value, "
    "or all values were DISCARDED by the screening steps:\n"
    "  <parameter_name> | not_extractable"
)


PARAM_META: dict[str, dict] = {
    "size_nm": {
        "description": "Nanoparticle diameter or size explicitly measured in the study",
        "unit_hint": "nm",
        "specific_format": "size_nm | <value> | <unit> | <drug_name> | <size_type>",
        "field_rules": {
            "<value>": "Numeric size exactly as reported (include ranges , intervals and deviations).",
            "<unit>": "Unit exactly as written in text.",
            "<drug_name>": (
                "Identify the formulation associated to the reported size."
                "If no formulation name is specified, write 'not extractable'."
            ),
            "<size_type>": (
                "Type of particle size measurement reported. "
                "Indicate the method used to determine size, such as 'DLS', "
                "'hydrodynamic diameter', 'TEM', 'z-average' "
                "or other explicitly stated measurement type. "
                "Use only values explicitly mentioned in the text; if not specified, write 'not extractable."
            )
    },
        "exclude": [
            "theoretical sizes",
            "expected sizes",
            "pore sizes",
            "filter sizes",
            "instrument limits",
        ],
    },

    "zeta_potential_mv": {
        "description": "Zeta potential (surface charge)",
        "unit_hint": "mV",
        "specific_format": "zeta_potential_mv | <value> | <unit>",
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
        "description": "Polydispersity index",
        "unit_hint": "dimensionless",
        "specific_format": "pdi | <value>| <methodology>",
        "field_rules": {
            "<value>": "Numeric PDI exactly as reported.",
            "<methodology>": "Preparation, formulation, or measurement method asssociated to the reported PDI",
        },
        "exclude": [],
    },

    "encapsulation_efficiency_pct": {
        "description": "Encapsulation efficiency or EE",
        "unit_hint": "%",
        "specific_format": "encapsulation_efficiency_pct | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric EE (Encapsulation Efficiency) exactly as reported (include the standard deviation if available).",
            "<unit>": "Percentage or unit as written.",
            "<drug_name>": "Drug explicitly mentioned in experiment.",
        },
        "exclude": [
            "theoretical efficiency",
        ],
    },

    "ic50": {
        "description": "Half-maximal inhibitory concentration",
        "unit_hint": "uM, nM, mg/mL, etc.",
        "specific_format": "ic50 | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric IC50 exactly as reported.",
            "<unit>": "Unit exactly as written.",
            "<drug_name>": " Formulation code/ drug name.",
        },
        "exclude": [],
    },

    "distribution_half_life_h": {
        "description": "Distribution half-life (alpha phase)",
        "unit_hint": "h",
        "specific_format": "distribution_half_life_h | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric half-life exactly as reported.",
            "<unit>": "Unit in hours or as stated.",
            "<drug_name>": "Formulation code associated to the nanoparticle.",
        },
        "exclude": [],
    },

    "circulation_half_life_h": {
        "description": "Circulation / elimination half-life",
        "unit_hint": "h",
        "specific_format": "circulation_half_life_h | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric half-life exactly as reported.",
            "<unit>": "Unit as written.",
            "<drug_name>": "Drug studied.",
        },
        "exclude": [],
    },

    "dose_group": {
        "description": "Administered treatment doses from the AUTHORS' OWN experiment only",
        "unit_hint": "as reported",
        "specific_format": "dose_group | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric only, exactly as reported. Do NOT include units or route.",
            "<unit>": "Dose unit exactly as written. Valid formats: mg/kg, μg/kg, mg/animal, μg, mg, g, etc.",
            "<drug_name>": "Drug actually administered in this study.",
        },
        "exclude": [
            "theoretical doses",
        ],
    },

    "tumor_reduction": {
        "description": "Tumor size reduction/ tumor reduction  vs control",
        "unit_hint": "%, mm^3, or as reported",
        "specific_format": "tumor_reduction | <value> | <unit> | <drug_name>",
        "field_rules": {
            "<value>": "Numeric reduction exactly as reported.",
            "<unit>": "Percentage.",
            "<drug_name>": "Drug or treatment used.",
        },
        "exclude": [
            "predicted inhibition",
        ],
    },

    "delivery_efficiency": {
        "description": "Cellular or in-vivo delivery efficiency",
        "unit_hint": "% or fold-change",
        "specific_format": "delivery_efficiency | <value> | <unit> | <condition>",
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
        "description": "Organ accumulation/ biodistribution",
        "unit_hint": "% ID, %ID/g, etc.",
        "specific_format": "biodistribution | <value> | <unit> | <organ>",
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
            "You are a nanoparticle information-extraction assistant."
            #"You extract data truthfully from scientific text. "
            #"Only extract values explicitly stated as part of the AUTHORS' OWN experiment. "
            #"Do not infer, guess, or hallucinate values."
        )

        # INITIALIZATION 
        initialization = (
            "Only extract values explicitly stated as part of the AUTHORS' OWN experiment."
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
            schema_lines.append("Format answer:")
            schema_lines.extend(override_lines)

        schema_lines.append("Rules:")
        schema_lines.append("- One line per extracted value, no NOT omit any field of the 'Format answer' given, ALWAYS include the parameter name.")
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
            "Format answer (include always all fields)": "\n".join(schema_lines),
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
    

class PromptCreationLipidComposition(BaseModel):

    def build_extraction_prompt_json(self, lipids_found: list[str]) -> dict:

        expertise = (
            "You are an expert assistant for identifying lipid components in "
            "nanoparticle formulations described in scientific text."
        )

        initialization = (
            "A first-pass scan already identified the following lipids/lipid-like "
            "components in the text below:\n"
            + (", ".join(lipids_found) if lipids_found else "(none)")
        )

        objective = (
            "Re-read the text and check whether any OTHER lipid, lipid derivative, "
            "sterol, PEG-lipid, or ionizable lipid is mentioned that is NOT already "
            "in the list above. Only consider compounds that are part of the "
            "nanoparticle's own lipid composition (used to formulate the particle), "
            "not unrelated excipients, drugs, or buffers."
        )

        schema_lines = [
            "Rules:",
            "- List ONLY lipids that are missing from the list above.",
            "- One lipid name per line, exactly as written in the text.",
            "- Do NOT repeat lipids already in the list above.",
            "- If nothing is missing, answer exactly: none",
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the missing lipid names (or 'none'). No explanations, headers, or comments.",
        }
    
class PromptCreationLoadStatus(BaseModel):

    def build_extraction_prompt_json(self, size_entries: list[dict]) -> dict:
        expertise = (
            "You are an expert assistant for determining the loading status of "
            "nanoparticle formulations described in scientific text."
        )
        initialization = (
            "Read the paragraph carefully."
        )
        objective = "Only define each formulation as loaded, unloaded, or unknown using the rules below."

        entries_str = "\n".join(
            f"- size={e.get('value')} {e.get('unit')}, drug={e.get('drug_name') or '(none)'}"
            for e in size_entries
        )

        schema_lines = [
            "Rules:",
            "- Use 'loaded' ONLY if the same sentence (or its immediate clause) explicitly "
            "states the particle is loaded/encapsulated/incorporated with a drug or cargo.",
            "- Use 'unloaded' ONLY if the same sentence (or its immediate clause) explicitly "
            "states the particle is 'blank', 'empty', 'unloaded', or 'control'.",
            "- Do NOT infer loading status from formulation codes "
            "mentioned earlier in the text unless that code or its explicit label is used "
            "in the same sentence as this size value.",
            "- Do NOT assume a value belongs to a different formulation just because a "
            "different preparation method (e.g. a different technique) is mentioned.",
            "- If unsure, answer 'unknown'.",
            "Format: <size_value> | <unit> | <status>",
            "Size values to classify:",
            entries_str,
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the classification lines. No explanations, headers, or comments.",
        }
    
class PromptCreationLipidRatioUnits(BaseModel):

    def build_extraction_prompt_json(self, ratio_entries: list[dict]) -> dict:
        expertise = (
            "You are an expert assistant for identifying the quantification type "
            "of lipid composition ratios in nanoparticle formulations."
        )

        entries_str = "\n".join(
            f"- {e.get('lipid')}: {e.get('ratio')}"
            for e in ratio_entries
        )

        initialization = (
            "The following lipid:ratio pairs were already identified in the text below:\n"
            + entries_str
        )

        objective = (
            "For the lipids ratio listed above, determine the quantification type "
            "explicitly used in the text (e.g. molar ratio, mol%, weight ratio, w/w, v/v)."
        )

        schema_lines = [
            "Format: lipid_composition_ratio_units | <quantification_type>",
            "Rules:",
            "- Use the quantification type EXACTLY as stated in the text near that ratio "
            "(e.g. 'mol%', 'molar ratio', 'w/w', 'weight ratio').",
            "- Do NOT infer or guess a unit; only use what is explicitly written.",
            "- If no quantification type is not written just replace by 'unknown'."
        ]

        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the extraction line. No explanations, headers, or comments.",
        }
    
class PromptCreationFormulationRegistry(BaseModel):

    def build_extraction_prompt_json(self, candidate_codes: list[str]) -> dict:
        expertise = (
            "You are an expert assistant for identifying what nanoparticle "
            "formulation codes or abbreviations refer to in scientific text."
        )
        initialization = (
            "A first-pass scan found these candidate formulation codes/abbreviations "
            "in the text below:\n" + ", ".join(candidate_codes)
        )
        objective = (
            "For each REAL formulation code (i.e. one that genuinely refers to a "
            "specific nanoparticle/liposome formulation in this text), determine: "
            "(1) the drug or cargo it contains, if any, and "
            "(2) whether it is explicitly described as loaded or unloaded/blank. "
            "Only use information explicitly stated in the text. Do not infer drug "
            "identity or load status from the code's letters/name alone "
            "(e.g. do not assume 'BLK' means blank just because of the abbreviation; "
            "only conclude this if the text itself states it)."
        )
        schema_lines = [
            "Format: <code> | <drug_name_or_none> | <loaded/unloaded/unknown>",
            "Rules:",
            "- If a candidate is NOT actually a formulation identifier (e.g. it's a "
            "method name, unit, or unrelated abbreviation), SKIP it — do not output a line for it.",
            "- Use 'none' for drug_name if no cargo is stated for that code.",
            "- Use 'unknown' for load status only if the text genuinely does not state it.",
            "- One line per valid formulation code.",
        ]
        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the extraction lines. Do not add any explanations, headers, or comments.",
        }
    
class PromptCreationCargoCategoryCheck(BaseModel):

    def build_extraction_prompt_json(self, unmatched_cargos: list[str], known_categories: list[str]) -> dict:
        expertise = (
            "You are an expert pharmacology assistant classifying therapeutic "
            "compounds into drug class categories."
        )
        initialization = (
            "The following candidate drug/cargo names were found in the text but "
            "are NOT present in our reference database:\n"
            + ", ".join(unmatched_cargos)
        )
        objective = (
            "For each name, determine if it is a real therapeutic compound, drug, "
            "or biologically active cargo (not a lipid, buffer, or excipient). "
            "If so, assign the closest matching category from the list below, or "
            "'new_category' if none fit well. If it is NOT a real drug/cargo, mark it 'not_a_drug'."
        )
        schema_lines = [
            "Known categories: " + ", ".join(known_categories),
            "Format: <name> | <is_drug: yes/no> | <category_or_new_category_or_not_applicable>",
            "Rules:",
            "- One line per name, in the same order given.",
            "- Do not invent a name not in the input list.",
        ]
        return {
            "expertise": expertise,
            "initialization": initialization,
            "definitions": {},
            "objective": objective,
            "answer_schema": {"Format": "\n".join(schema_lines)},
            "conclusion": "Return ONLY the extraction lines. No explanations, headers, or comments.",
        }

