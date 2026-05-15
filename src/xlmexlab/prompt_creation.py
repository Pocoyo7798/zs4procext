"""
Parameter extraction pipeline:
  1. build_extraction_prompt_json() → creates the LLM prompt dict for a paragraph,
     with keys matching PromptFormatter fields exactly.
  2. parse_extraction_response()    → parses the LLM key-value output into structured dicts.

PromptFormatter field → chat-template turn mapping:

  SYSTEM turn (stable, same for every paragraph):
    expertise       → model identity
    initialization  → hard constraint: only extract explicit values

  USER turn (dynamic, rebuilt per paragraph):
    definitions     → parameter list with descriptions and expected units
    objective       → the task in one short sentence
    answer_schema   → output format rules (default + overrides)
    context         → the paragraph  [passed via format_prompt(context=paragraph)]
    conclusion      → short closing reminder (no extra text)

  Template order:
    <|im_start|>system
    {expertise}{initialization}<|im_end|>
    <|im_start|>user
    {definitions}{objective}{answer_schema}{context}{conclusion}<|im_end|>
    <|im_start|>assistant
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
        "unit_hint": "dimensionless (0-1)",
    },
    "encapsulation_efficiency_pct": {
        "description": "Encapsulation efficiency",
        "unit_hint": "%",
    },
    "ic50": {
        "description": "Half-maximal inhibitory concentration (IC50)",
        "unit_hint": "uM, nM, mg/mL, or as reported",
    },
    "distribution_half_life_h": {
        "description": "Distribution half-life (alpha phase)",
        "unit_hint": "h",
    },
    "circulation_half_life_h": {
        "description": "Circulation / elimination half-life",
        "unit_hint": "h",
    },
    "dose_group": {
        "description": "Administered dose",
        "unit_hint": "mg/kg, mg/m2, ug or as reported",
        "specific_format": "<parameter_name> | <value> | <unit> | <drug_name> | <schedule>",
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

# Default output format applied to all parameters unless overridden by specific_format
_DEFAULT_FORMAT = "<parameter_name> | <value> | <unit> | <condition>"


class PromptCreation(BaseModel):

    def build_extraction_prompt_json(
        self,
        paragraph: str,
        extracted_flags: dict[str, Any],
    ) -> tuple[dict, list[str]]:
        """
        Build a prompt dict whose keys map directly to PromptFormatter fields.

        Returns:
            prompt_json : dict  – ready to unpack into PromptFormatter(**prompt_json)
            targets     : list  – active parameter names
        """
        targets = [k for k, v in extracted_flags.items() if v is True]

        if not targets:
            return {}, []

        # ------------------------------------------------------------------
        # SYSTEM TURN — stable across all paragraphs
        # ------------------------------------------------------------------
        expertise = (
            "You are a nanoparticle information-extraction assistant. "
            "You extract data truthfully from scientific text."
        )

        initialization = (
            "Only extract values explicitly stated in the paragraph. "
            "Do not infer, guess, or hallucinate values."
        )

        # ------------------------------------------------------------------
        # USER TURN — rebuilt per paragraph
        # ------------------------------------------------------------------

        # 1. definitions — what each parameter is and its expected unit
        definitions: dict[str, str] = {
            param: (
                f"{PARAM_META[param].get('description', param.replace('_', ' '))}. "
                f"Expected unit: {PARAM_META[param].get('unit_hint', 'as reported')}."
            )
            for param in targets
        }

        # 2. objective — the task, one sentence
        objective = "Extract ONLY the parameters listed above from the paragraph below."

        # 3. answer_schema — all format rules in one place
        #    3a. default format line
        schema_lines: list[str] = [
            f"Default format per line: {_DEFAULT_FORMAT}",
        ]

        #    3b. parameter-specific format overrides (if any)
        override_lines: list[str] = [
            f"  - {param}: {PARAM_META[param]['specific_format']}"
            for param in targets
            if "specific_format" in PARAM_META.get(param, {})
        ]
        if override_lines:
            schema_lines.append("Format overrides:")
            schema_lines.extend(override_lines)

        #    3c. field-level rules
        schema_lines += [
            "Rules:",
            "- One line per value. If a parameter has values under different conditions",
            "  (e.g. pH, temperature, time-point, cell line, formulation),",
            "  output one line per condition.",
            "- <condition>: the condition for that value, or 'none' if absent.",
            "- <value>: numeric, exactly as reported, including deviations and ranges",
            "  (e.g. 155 +/- 0.1, -18.3, 0.12-0.14, >100, <=200).",
            "- <unit>: exactly as reported in the text.",
            "- If a parameter is mentioned but its value is not numeric,",
            "  write: <parameter_name> NOT EXTRACTABLE",
        ]

        answer_schema: dict[str, str] = {
            "Format": "\n".join(schema_lines)
        }

        # 4. conclusion — single closing reminder, keeps the model on track
        conclusion = "Return ONLY the extraction lines. No explanations, headers, or comments."

        # context (the paragraph) is NOT included here —
        # it is passed separately via format_prompt(context=paragraph)
        prompt_json = {
            "expertise":      expertise,
            "initialization": initialization,
            "definitions":    definitions,
            "objective":      objective,
            "answer_schema":  answer_schema,
            "conclusion":     conclusion,
        }

        return prompt_json, targets