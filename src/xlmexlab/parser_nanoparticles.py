#parser
import re
from collections import OrderedDict
from pydantic import BaseModel, PrivateAttr
import Levenshtein
from typing import List

CARGO_DB = OrderedDict({

    "mrna": [
        "neoantigen mRNA", "mRNA-4157", "mRNA-4359",
        "OX40L mRNA", "IL-12 mRNA", "IL-23 mRNA",
        "TGF-β trap mRNA", "WT1 antigen mRNA",
        "HER2 antigen mRNA", "MUC1 mRNA",
        "TRP2 mRNA", "gp100 mRNA", "p53 mRNA",
        "Cas9 mRNA", "Cas12a mRNA"
    ],

    "sarna_circrna": [
        "MUC1 saRNA", "HER2 saRNA",
        "NY-ESO-1 saRNA", "hTERT circRNA"
    ],

    "sirna_shrna": [
        "STAT3 siRNA", "survivin siRNA", "BCL-2 siRNA",
        "VEGF siRNA", "MMP-9 siRNA", "EZH2 siRNA",
        "Twist1 siRNA", "EGFR siRNA", "anti-EGFR siRNA",
        "PI3K siRNA", "AKT siRNA", "generic siRNA", "shRNA"
    ],

    "mirna": [
        "miR-34a", "miR-155", "miR-21", "miR-200",
        "miR-145", "miR-10b", "miR-373",
        "miR-182-3p", "let-7"
    ],

    "crispr": [
        "Cas9 + HER2 sgRNA",
        "Cas9 + ESR1 sgRNA",
        "Cas9 + PIK3CA sgRNA",
        "base editor BRCA1",
        "sgRNA", "crRNA"
    ],

    "chemotherapy": [
        "paclitaxel", "docetaxel", "DOX",
        "epirubicin", "gemcitabine", "5-FU",
        "eribulin", "mertansine"
    ],

    "targeted_small_molecules": [
        "lapatinib", "olaparib",
        "bicalutamide", "rapamycin", "disulfiram"
    ],

    "bisphosphonates": [
        "zoledronic acid", "ZOL",
        "Man-LP@ZOL", "Man-NP@ZOL", "alendronate"
    ],

    "immune_modulators": [
        "imiquimod", "R848", "CpG ODN",
        "poly I:C", "STING agonist", "cGAMP",
        "DMXAA", "TRAIL", "TLR9 agonist"
    ],

    "antibodies_peptides": [
        "trastuzumab", "anti-PD-L1",
        "RGD peptide", "iRGD",
        "C-peptide-SLN-PTX"
    ],

    "imaging_phototherapy": [
        "ICG", "Ce6", "BPD", "Gd-DTPA"
    ],

    "natural_products": [
        "curcumin", "resveratrol",
        "ginsenoside", "quercetin",
        "EGCG", "BER"
    ]
})

cargo_list = sorted({
    x for vals in CARGO_DB.values() for x in vals
})


SEP = r"[-_\s/]+"          # flexible separator
RNA = r"(?:rna)?"
WORD = r"\b"

def rx(term):
    return WORD + term + WORD


# 3. LITERATURE-SCALE NOMENCLATURE MAP


cargo_map = OrderedDict({
    # mRNA / Vaccines
    rx(r"neoantigen(?:specific)?"+SEP+r"m"+RNA): "neoantigen mRNA",
    rx(r"personali[sz]ed"+SEP+r"tumou?r"+SEP+r"antigens?"+SEP+r"m"+RNA): "neoantigen mRNA",

    rx(r"m"+RNA+SEP+r"4157"): "mRNA-4157",
    rx(r"mrna4157"): "mRNA-4157",

    rx(r"m"+RNA+SEP+r"4359"): "mRNA-4359",
    rx(r"mrna4359"): "mRNA-4359",

    rx(r"ox40l"+SEP+r"m"+RNA): "OX40L mRNA",
    rx(r"cd134l"+SEP+r"m"+RNA): "OX40L mRNA",

    rx(r"il"+SEP+r"12"+SEP+r"m"+RNA): "IL-12 mRNA",
    rx(r"interleukin"+SEP+r"12"+SEP+r"m"+RNA): "IL-12 mRNA",

    rx(r"il"+SEP+r"23"+SEP+r"m"+RNA): "IL-23 mRNA",
    rx(r"interleukin"+SEP+r"23"+SEP+r"m"+RNA): "IL-23 mRNA",

    rx(r"tgf"+SEP+r"[βb]"+SEP+r"trap"+SEP+r"m"+RNA): "TGF-β trap mRNA",
    rx(r"transforming"+SEP+r"growth"+SEP+r"factor"+SEP+r"beta"+SEP+r"trap"): "TGF-β trap mRNA",

    rx(r"ro7198457"): "WT1 antigen mRNA",
    rx(r"wt1"+SEP+r"(antigen)?"+SEP+r"m"+RNA): "WT1 antigen mRNA",

    rx(r"bnt111"): "HER2 antigen mRNA",
    rx(r"her2"+SEP+r"(antigen)?"+SEP+r"m"+RNA): "HER2 antigen mRNA",
    rx(r"erbb2"+SEP+r"m"+RNA): "HER2 antigen mRNA",

    rx(r"muc1"+SEP+r"m"+RNA): "MUC1 mRNA",
    rx(r"trp2"+SEP+r"m"+RNA): "TRP2 mRNA",
    rx(r"gp100"+SEP+r"m"+RNA): "gp100 mRNA",
    rx(r"p53"+SEP+r"m"+RNA): "p53 mRNA",
    rx(r"tp53"+SEP+r"m"+RNA): "p53 mRNA",

    rx(r"cas9"+SEP+r"m"+RNA): "Cas9 mRNA",
    rx(r"spcas9"+SEP+r"m"+RNA): "Cas9 mRNA",

    rx(r"cas12a"+SEP+r"m"+RNA): "Cas12a mRNA",
    rx(r"cpf1"+SEP+r"m"+RNA): "Cas12a mRNA",

    
    # saRNA / circRNA
    

    rx(r"muc1"+SEP+r"sa"+RNA): "MUC1 saRNA",
    rx(r"her2"+SEP+r"sa"+RNA): "HER2 saRNA",
    rx(r"ny"+SEP+r"eso"+SEP+r"1"+SEP+r"sa"+RNA): "NY-ESO-1 saRNA",
    rx(r"htert"+SEP+r"circ"+RNA): "hTERT circRNA",
    rx(r"circular"+SEP+r"rna"+SEP+r"htert"): "hTERT circRNA",

    
    # siRNA / shRNA
    

    rx(r"anti"+SEP+r"egfr"+SEP+r"si"+RNA): "anti-EGFR siRNA",
    rx(r"egfr"+SEP+r"si"+RNA): "EGFR siRNA",
    rx(r"erbb1"+SEP+r"si"+RNA): "EGFR siRNA",

    rx(r"stat3"+SEP+r"si"+RNA): "STAT3 siRNA",
    rx(r"survivin"+SEP+r"si"+RNA): "survivin siRNA",
    rx(r"birc5"+SEP+r"si"+RNA): "survivin siRNA",

    rx(r"bcl"+SEP+r"2"+SEP+r"si"+RNA): "BCL-2 siRNA",
    rx(r"bcl2"+SEP+r"si"+RNA): "BCL-2 siRNA",

    rx(r"vegf"+SEP+r"si"+RNA): "VEGF siRNA",
    rx(r"vegfa"+SEP+r"si"+RNA): "VEGF siRNA",

    rx(r"mmp"+SEP+r"9"+SEP+r"si"+RNA): "MMP-9 siRNA",
    rx(r"ezh2"+SEP+r"si"+RNA): "EZH2 siRNA",
    rx(r"twist1"+SEP+r"si"+RNA): "Twist1 siRNA",
    rx(r"pi3k"+SEP+r"si"+RNA): "PI3K siRNA",
    rx(r"pik3ca"+SEP+r"si"+RNA): "PI3K siRNA",
    rx(r"akt"+SEP+r"si"+RNA): "AKT siRNA",
    rx(r"akt1"+SEP+r"si"+RNA): "AKT siRNA",

    rx(r"short"+SEP+r"hairpin"+SEP+r"rna"): "shRNA",
    rx(r"sh"+RNA): "shRNA",

    rx(r"small"+SEP+r"interfering"+SEP+r"rna"): "generic siRNA",
    rx(r"si"+RNA): "generic siRNA",

    
    # miRNA
    

    rx(r"mir"+SEP+r"34a"): "miR-34a",
    rx(r"micro"+SEP+r"rna"+SEP+r"34a"): "miR-34a",

    rx(r"mir"+SEP+r"155"): "miR-155",
    rx(r"mir"+SEP+r"21"): "miR-21",
    rx(r"mir"+SEP+r"200"): "miR-200",
    rx(r"mir"+SEP+r"145"): "miR-145",
    rx(r"mir"+SEP+r"10b"): "miR-10b",
    rx(r"mir"+SEP+r"373"): "miR-373",
    rx(r"mir"+SEP+r"182"+SEP+r"3p"): "miR-182-3p",

    rx(r"let"+SEP+r"7"): "let-7",

    
    # CRISPR / guides
    

    rx(r"cas9"+SEP+r"her2"+SEP+r"sg"+RNA): "Cas9 + HER2 sgRNA",
    rx(r"cas9"+SEP+r"esr1"+SEP+r"sg"+RNA): "Cas9 + ESR1 sgRNA",
    rx(r"cas9"+SEP+r"pik3ca"+SEP+r"sg"+RNA): "Cas9 + PIK3CA sgRNA",

    rx(r"single"+SEP+r"guide"+SEP+r"rna"): "sgRNA",
    rx(r"sg"+RNA): "sgRNA",

    rx(r"crispr"+SEP+r"rna"): "crRNA",
    rx(r"cr"+RNA): "crRNA",

    rx(r"base"+SEP+r"editor"+SEP+r"brca1"): "base editor BRCA1",

    
    # Chemotherapy
    

    rx(r"paclitaxel"): "paclitaxel",
    rx(r"taxol"): "paclitaxel",
    rx(r"ptx"): "paclitaxel",

    rx(r"docetaxel"): "docetaxel",
    rx(r"taxotere"): "docetaxel",

    rx(r"dox"): "DOX",
    rx(r"doxorubicin"): "DOX",
    rx(r"adriamycin"): "DOX",
    rx(r"hydroxydaunorubicin"): "DOX",

    rx(r"epirubicin"): "epirubicin",
    rx(r"gemcitabine"): "gemcitabine",
    rx(r"5"+SEP+r"fu"): "5-FU",
    rx(r"fluorouracil"): "5-FU",

    rx(r"eribulin"): "eribulin",
    rx(r"halaven"): "eribulin",

    rx(r"mertansine"): "mertansine",
    rx(r"dm1"): "mertansine",

    
    # Targeted / Small molecules
    

    rx(r"lapatinib"): "lapatinib",
    rx(r"tykerb"): "lapatinib",

    rx(r"olaparib"): "olaparib",
    rx(r"lynparza"): "olaparib",

    rx(r"bicalutamide"): "bicalutamide",
    rx(r"casodex"): "bicalutamide",

    rx(r"rapamycin"): "rapamycin",
    rx(r"sirolimus"): "rapamycin",
    rx(r"rap"): "rapamycin",

    rx(r"disulfiram"): "disulfiram",
    rx(r"ds"): "disulfiram",
    rx(r"antabuse"): "disulfiram",

    
    # Bisphosphonates
    

    rx(r"man"+SEP+r"lp@zol"): "Man-LP@ZOL",
    rx(r"mannosylated"+SEP+r"liposome"+SEP+r"zol"): "Man-LP@ZOL",

    rx(r"man"+SEP+r"np@zol"): "Man-NP@ZOL",

    rx(r"zoledronic"+SEP+r"acid"): "zoledronic acid",
    rx(r"zoledronate"): "zoledronic acid",
    rx(r"zometa"): "zoledronic acid",
    rx(r"zol"): "ZOL",

    rx(r"alendronate"): "alendronate",
    rx(r"fosamax"): "alendronate",

    
    # Immunomodulators
    

    rx(r"imiquimod"): "imiquimod",
    rx(r"r837"): "imiquimod",

    rx(r"r848"): "R848",
    rx(r"resiquimod"): "R848",

    rx(r"cpg"+SEP+r"odn"): "CpG ODN",
    rx(r"cpg"+SEP+r"oligodeoxynucleotide"): "CpG ODN",

    rx(r"poly"+SEP+r"i:c"): "poly I:C",
    rx(r"polyinosinic"+SEP+r"polycytidylic"+SEP+r"acid"): "poly I:C",

    rx(r"sting"+SEP+r"agonist"): "STING agonist",
    rx(r"stimulator"+SEP+r"of"+SEP+r"interferon"+SEP+r"genes"): "STING agonist",

    rx(r"cgamp"): "cGAMP",
    rx(r"2'?3'?"+SEP+r"cgamp"): "cGAMP",

    rx(r"dmxaa"): "DMXAA",
    rx(r"vadimezan"): "DMXAA",

    rx(r"trail"): "TRAIL",
    rx(r"tnf"+SEP+r"related"+SEP+r"apoptosis"+SEP+r"inducing"+SEP+r"ligand"): "TRAIL",

    rx(r"tlr9"): "TLR9 agonist",
    rx(r"toll"+SEP+r"like"+SEP+r"receptor"+SEP+r"9"): "TLR9 agonist",

    
    # Antibodies / Peptides
    

    rx(r"trastuzumab"): "trastuzumab",
    rx(r"herceptin"): "trastuzumab",

    rx(r"anti"+SEP+r"pd"+SEP+r"l1"): "anti-PD-L1",
    rx(r"pd"+SEP+r"l1"+SEP+r"antibody"): "anti-PD-L1",

    rx(r"rgd"+SEP+r"peptide"): "RGD peptide",
    rx(r"arg"+SEP+r"gly"+SEP+r"asp"): "RGD peptide",

    rx(r"irgd"): "iRGD",

    rx(r"c"+SEP+r"peptide"+SEP+r"sln"+SEP+r"ptx"): "C-peptide-SLN-PTX",

    
    # Imaging / Phototherapy
    

    rx(r"icg"): "ICG",
    rx(r"indocyanine"+SEP+r"green"): "ICG",

    rx(r"ce6"): "Ce6",
    rx(r"chlorin"+SEP+r"e6"): "Ce6",

    rx(r"bpd"): "BPD",
    rx(r"benzoporphyrin"): "BPD",
    rx(r"benzoporphyrin"+SEP+r"derivative"): "BPD",
    rx(r"verteporfin"): "BPD",

    rx(r"gd(?:3\+|\u00B3\+)?"+SEP+r"dtpa"): "Gd-DTPA",
    rx(r"gadolinium"+SEP+r"dtpa"): "Gd-DTPA",
    rx(r"magnevist"): "Gd-DTPA",

    
    # Natural products
    

    rx(r"curcumin"): "curcumin",
    rx(r"diferuloylmethane"): "curcumin",

    rx(r"resveratrol"): "resveratrol",

    rx(r"ginsenoside"): "ginsenoside",

    rx(r"quercetin"): "quercetin",

    rx(r"egcg"): "EGCG",
    rx(r"epigallocatechin"+SEP+r"gallate"): "EGCG",

    rx(r"ber"): "BER",
    rx(r"berberine"): "BER",
})

# 4. COMPILE
COMPILED_MAP = [
    (re.compile(pattern, re.I), canonical)
    for pattern, canonical in cargo_map.items()
]


ORGAN_MAP = {
    # Liver
    "liver": "liver",
    "hepatic": "liver",
    "hepatocyte": "liver",
    "hepatocellular": "liver",

    # Spleen
    "spleen": "spleen",
    "splenic": "spleen",

    # Kidney
    "kidney": "kidney",
    "renal": "kidney",
    "nephric": "kidney",

    # Lung
    "lung": "lung",
    "pulmonary": "lung",
    "bronchus": "lung",
    "bronchial": "lung",
    "alveolar": "lung",

    # Heart
    "heart": "heart",
    "cardiac": "heart",
    "myocardial": "heart",

    # Brain / CNS
    "brain": "brain",
    "cerebral": "brain",
    "cortex": "brain",
    "hippocampus": "brain",
    "hypothalamus": "brain",
    "pineal": "brain",
    "pituitary": "brain",
    "pituitary gland": "brain",
    "spinal cord": "brain",
    "neural": "brain",
    "nerve": "brain",

    # Intestine / GI
    "intestine": "intestine",
    "intestinal": "intestine",
    "gut": "intestine",
    "bowel": "intestine",
    "duodenum": "intestine",
    "jejunum": "intestine",
    "ileum": "intestine",
    "colon": "intestine",
    "colonic": "intestine",
    "rectum": "intestine",
    "rectal": "intestine",
    "appendix": "intestine",
    "appendiceal": "intestine",

    # Stomach
    "stomach": "stomach",
    "gastric": "stomach",

    # Pancreas
    "pancreas": "pancreas",
    "pancreatic": "pancreas",

    # Gallbladder / biliary
    "gallbladder": "gallbladder",
    "biliary": "gallbladder",
    "bile duct": "gallbladder",

    # Bladder
    "bladder": "bladder",
    "urinary bladder": "bladder",

    # Blood / circulation
    "blood": "blood",
    "plasma": "blood",
    "serum": "blood",
    "vascular": "blood vessel",
    "vessel": "blood vessel",
    "aorta": "blood vessel",
    "artery": "blood vessel",
    "arterial": "blood vessel",
    "vein": "blood vessel",
    "venous": "blood vessel",
    "capillary": "blood vessel",

    # Lymphatic
    "lymph node": "lymph node",
    "lymphatic": "lymph node",
    "thymus": "thymus",
    "tonsil": "tonsil",
    "tonsillar": "tonsil",

    # Reproductive male
    "prostate": "prostate",
    "prostatic": "prostate",
    "testis": "testis",
    "testes": "testis",
    "testicular": "testis",

    # Reproductive female
    "ovary": "ovary",
    "ovarian": "ovary",
    "uterus": "uterus",
    "uterine": "uterus",
    "cervix": "cervix",
    "cervical": "cervix",
    "vagina": "vagina",
    "vaginal": "vagina",
    "breast": "breast",
    "mammary": "breast",

    # Endocrine
    "thyroid gland": "thyroid",
    "thyroid": "thyroid",
    "adrenal": "adrenal gland",
    "adrenal gland": "adrenal gland",

    # Head / ENT
    "eye": "eye",
    "ocular": "eye",
    "retina": "eye",
    "ear": "ear",
    "otic": "ear",
    "nose": "nose",
    "nasal": "nose",
    "sinus": "nose",
    "pharynx": "throat",
    "larynx": "throat",
    "trachea": "throat",
    "esophagus": "esophagus",
    "esophageal": "esophagus",

    # Skin / soft tissue
    "skin": "skin",
    "cutaneous": "skin",
    "muscle": "muscle",
    "skeletal muscle": "muscle",
    "fat": "adipose tissue",
    "adipose": "adipose tissue",
    "soft tissue": "soft tissue",
    "cartilage": "cartilage",
    "tendon": "tendon",
    "ligament": "ligament",
    "joint": "joint",
    "synovial": "joint",
    "bone marrow": "bone marrow",
    "bone": "bone",

    # Body cavities / membranes
    "peritoneum": "peritoneum",
    "peritoneal": "peritoneum",
    "pleura": "pleura",
    "pleural": "pleura",
    "pericardium": "pericardium",
    "pericardial": "pericardium",
    "diaphragm": "diaphragm",

    # Tumor / disease tissue
    "tumor": "tumor",
    "tumour": "tumor",
    "xenograft": "tumor",
    "metastasis": "tumor",
}

class ParserNanoparticle(BaseModel):
    _parameters: List[str] = PrivateAttr(default_factory=list)

    def correct_param(self, param:str, parameters: List[str])-> str:
        best_match = param
        best_score = 0

        for p in parameters:
            score = Levenshtein.ratio(param, p)

            if score > best_score:
                best_score = score
                best_match = p

        if best_score == 1:
            return param  # já é perfeito
        else:
            return best_match  # substitui pelo mais próximo
    def parse_extraction_response(self, response: str) -> dict[str, list[dict]]:
        """
        Parse the LLM key-value response into a structured dict.
    
        Returns:
            {
            "size_nm": [
                {"value": 155.0, "unit": "nm", "condition": "pH 7.4"},
                ...
            ],
            ...
            }
    
        Values that could not be extracted are stored as:
            {"value": None, "unit": None, "condition": None, "raw": "<original line>"}
        """
        results: dict[str, list[dict]] = {}
    
        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
    
            parts = [p.strip() for p in line.split("|")]
    
            # Expect exactly 4 parts; be lenient with trailing missing fields
            if len(parts) < 2:
                continue  # unrecognisable line
            
    
            param = parts[0]
            param = self.correct_param(param, self._parameters)
            value_str = parts[1] if len(parts) > 1 else ""
            unit = parts[2] if len(parts) > 2 else ""
            condition = parts[3] if len(parts) > 3 else "none"

            # Normalise sentinel values
            if unit in ("-", ""):
                unit = None
            if condition in ("-", "", "none"):
                condition = None
    
            # Try to cast value to float
            if value_str == "not_extractable" or value_str in ("-", ""):
                entry = {"value": None, "unit": unit, "condition": condition, "raw": line}
            else:
                numeric = value_str
                entry = {
                    "value": numeric if numeric is not None else value_str,
                    "unit": unit,
                    "condition": condition,
                }
                if numeric is None:
                    entry["raw"] = line  # keep original for debugging
    
            results.setdefault(param, []).append(entry)
    
        return results

    def detect_organ(self, text: str, organ_map: dict) -> str:
        text_low = text.lower()

        # ordenar por comprimento (multi-word first)
        keys = sorted(organ_map.keys(), key=len, reverse=True)

        for kw in keys:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_low):
                return organ_map[kw]

        return None


    def postprocess_biodistribution(self, entries: list[dict]) -> dict:
        processed = []

        for item in entries:
            raw_val = str(item.get("value", "")).replace("%", "").strip()

            try:
                percent = float(raw_val)
            except:
                percent = None

            cond = item.get("condition") or ""
            
            organ = self.detect_organ(cond, ORGAN_MAP) or "unknown"
            processed.append({
                "organ": organ,
                "percent": percent,
                "unit": item.get("unit"),
            })
        processed = sorted(
            processed,
            key=lambda x: (x["percent"] is not None, x["percent"]),
            reverse=True
        )

        final = {}
        for i, row in enumerate(processed, start=1):
            final[f"of_target_{i}"] = row

        return final

    def extract_cargos(self, text: str)-> str:
        """Return all unique cargos in order found."""
        for pat, name in COMPILED_MAP:
            if pat.search(text):
                if "free" in text:
                    condition = f'free {name}'
                    return condition
                else:
                    condition = name
                    return condition
            else: 
                    None
        return "unknown"


    def postprocess_IC50(self, entries: list[dict]) -> dict:
        processed = []
        for item in entries:

            cond = item.get("condition") or ""
            cargo = self.extract_cargos(cond) or "unknown"

            processed.append({
                "value": item.get("value"),
                "unit": item.get("unit"),
                "cargo": cargo,
            })
        return processed

    def postprocess_dose_group(self, entries: list[dict]) -> list[dict]:
        processed = []

        for item in entries:
            unit = item.get("unit")
            value = item.get("value")

            if unit == "mg/kg":
                try:
                    value = str(float(value) * 30)
                    unit = "µg"
                except (TypeError, ValueError):
                    pass

            processed.append({
                "value": value,
                "unit": unit,
                "condition": item.get("condition"),
            })

        return processed
    
    def compute_charge_group(self, entries: list[dict]) -> str | None:
        if not entries:
            return None

        try:
            # take first valid numeric value
            for item in entries:
                val = item.get("value")

                if val is None:
                    continue

                zeta = float(str(val).replace(",", ".").split()[0])

                if zeta > 10:
                    return "positive"
                elif zeta < -10:
                    return "negative"
                else:
                    return "neutral"

        except Exception:
            return None

        return None

    def replace(self, T_and_F_list: list[dict], simulated_llm_response: str) -> list[dict]:
        parsed = self.parse_extraction_response(simulated_llm_response)

        if "biodistribution" in parsed:
            parsed["biodistribution"] = self.postprocess_biodistribution(
                parsed["biodistribution"]
            )
        if "ic50" in parsed:
            parsed["ic50"] = self.postprocess_IC50(
                parsed["ic50"]
            )
        if "dose_group" in parsed:
            parsed["dose_group"] = self.postprocess_dose_group(
                parsed["dose_group"]
            )
        if "size_nm" in T_and_F_list and "size_nm" in parsed:
            T_and_F_list["size_nm"] = parsed["size_nm"]
        if "zeta_potential_mv" in T_and_F_list and "zeta_potential_mv" in parsed:
            T_and_F_list["zeta_potential_mv"] = parsed["zeta_potential_mv"]   
        if ("zeta_potential_mv" in T_and_F_list) and ("charge_group" in T_and_F_list is None):
            charge = self.compute_charge_group(T_and_F_list["zeta_potential_mv"])
            T_and_F_list["charge_group"] = charge
        if "pdi" in T_and_F_list and "pdi" in parsed:
            T_and_F_list["pdi"] = parsed["pdi"]
        if "encapsulation_efficiency_pct" in T_and_F_list and "encapsulation_efficiency_pct" in parsed:
            T_and_F_list["encapsulation_efficiency_pct"] = parsed["encapsulation_efficiency_pct"]
        if "ic50" in T_and_F_list and "ic50" in parsed:
            T_and_F_list["ic50"] = parsed["ic50"]
        if "distribution_half_life_h" in T_and_F_list and "distribution_half_life_h" in parsed:
            T_and_F_list["distribution_half_life_h"] = parsed["distribution_half_life_h"]             
        if "circulation_half_life_h" in T_and_F_list and "circulation_half_life_h" in parsed:
            T_and_F_list["circulation_half_life_h"] = parsed["circulation_half_life_h"]
        if "dose_group" in T_and_F_list and "dose_group" in parsed:
            T_and_F_list["dose_group"] = parsed["dose_group"]
        if "tumor_vol_reduction_pct" in T_and_F_list and "tumor_vol_reduction_pct" in parsed:
            T_and_F_list["tumor_vol_reduction_pct"] = parsed["tumor_vol_reduction_pct"]
        if "delivery_efficiency" in T_and_F_list and "delivery_efficiency" in parsed:
            T_and_F_list["delivery_efficiency"] = parsed["delivery_efficiency"]
        if "biodistribution" in T_and_F_list and "biodistribution" in parsed:
            T_and_F_list["biodistribution"] = parsed["biodistribution"]
        return T_and_F_list


