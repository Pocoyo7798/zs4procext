"""
NanoparticleExtractor
=====================
Hybrid extraction pipeline for nanoparticle study parameters.

Extraction strategy per feature type:
  - ~42 features  → regex / KeywordSearching / ParametersParser / MolarRatioFinder  (0 LLM calls)
  -  1 feature    → paragraph_classifier  (select relevant paper sections first)
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from dataclasses import is_dataclass, asdict
import json

from pydantic import BaseModel, PrivateAttr

from xlmexlab.nanoparticle_data import NanoparticleData

from collections import OrderedDict

from pathlib import Path
import pandas as pd



# 1. CANONICAL DATABASE
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
        "paclitaxel", "docetaxel", "doxorubicin",
        "epirubicin", "gemcitabine", "fluorouracil",
        "eribulin", "mertansine",
        "sacituzumab govitecan", "vinorelbine",                
    ],

    "targeted_small_molecules": [
        "lapatinib", "olaparib",
        "bicalutamide", "rapamycin", "disulfiram",
        "talazoparib", "palbociclib", "abemaciclib", "ribociclib",
        "erlotinib", "afatinib", "osimertinib",
        "crizotinib", "ceritinib", "alectinib", "brigatinib",
        "alpelisib", "copanlisib", "duvelisib",
        "ibrutinib", "acalabrutinib", "zanubrutinib"
    ],

    "bisphosphonates": [
        "zoledronic acid", "ZOL",
        "Man-LP@ZOL", "Man-NP@ZOL", "alendronate"
    ],

    "immune_modulators": [
        "imiquimod", "R848", "CpG Oligodeoxynucleotide",
        "poly I:C", "STING agonist", "cGAMP",
        "vadimezan", "dulanermin", "TLR9 agonist"   # DMXAA→vadimezan, TRAIL→dulanermin
    ],

    "antibodies_peptides": [
        "trastuzumab", "anti-PD-L1",
        "RGD peptide", "iRGD",
        "C-peptide-SLN-PTX"
    ],

    "checkpoint_inhibitors": [              # new category (mirrors Excel)
        "keytruda", "opdivo", "tecentriq",
        "yervoy", "avelumab", "durvalumab"
    ],

    "imaging_phototherapy": [
        "ICG", "Ce6", "BPD", "Gd-DTPA"
    ],

    "natural_products": [
        "curcumin", "resveratrol",
        "ginsenoside", "quercetin",
        "EGCG", "berberine", "lupeol",
    ],

    "hematopoietic_growth_factors": [
        "filgrastim", "pegfilgrastim",
        "lenograstim"
    ],   
})

cargo_list = sorted({
    x for vals in CARGO_DB.values() for x in vals
})


# 2. HELPERS
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
    rx(r"nab-paclitaxel"): "paclitaxel",
    rx(r"abraxane"): "paclitaxel",
    rx(r"vinorelbine"): "vinorelbine",
    rx(r"navelbine"): "vinorelbine", 
    rx(r"vinorelbel"): "vinorelbine",
    rx(r"binorel"): "vinorelbine",
    rx(r"biovelbin"): "vinorelbine",
    rx(r"eunades"): "vinorelbine",
    rx(r"flonorbin"): "vinorelbine",
    rx(r"neoben"): "vinorelbine",
    rx(r"relbovin"): "vinorelbine",
    rx(r"vinelbine"): "vinorelbine",
    rx(r"vinotec"): "vinorelbine",

    
    rx(r"docetaxel"): "docetaxel",
    rx(r"taxotere"): "docetaxel",

    rx(r"dox"): "doxorubicin",
    rx(r"doxorubicin"): "doxorubicin",
    rx(r"adriamycin"): "doxorubicin",
    rx(r"hydroxydaunorubicin"): "doxorubicin",
    rx(r"doxil"): "doxorubicin",

    rx(r"epirubicin"): "epirubicin",
    rx(r"gemcitabine"): "gemcitabine",

    rx(r"5"+SEP+r"fu"): "fluorouracil", 
    rx(r"5"+SEP+r"fluorouracil"): "fluorouracil",
    rx(r"fluorouracil"): "fluorouracil",

    rx(r"eribulin"): "eribulin",
    rx(r"halaven"): "eribulin",

    rx(r"mertansine"): "mertansine",
    rx(r"dm1"): "mertansine",

    rx(r"todelvy"): "sacituzumab govitecan",
    rx(r"trodelvy"): "sacituzumab govitecan",
    rx(r"sacituzumab"+SEP+r"govitecan"): "sacituzumab govitecan",

    
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

    rx(r"talazoparib"): "talazoparib",
    rx(r"talzenna"): "talazoparib",
 
    rx(r"palbociclib"): "palbociclib",
    rx(r"ibrance"): "palbociclib",
 
    rx(r"abemaciclib"): "abemaciclib",
    rx(r"verzenio"): "abemaciclib",
 
    rx(r"ribociclib"): "ribociclib",
    rx(r"kisqali"): "ribociclib",
 
    rx(r"erlotinib"): "erlotinib",
    rx(r"tarceva"): "erlotinib",
 
    rx(r"afatinib"): "afatinib",
    rx(r"gilotrif"): "afatinib",
 
    rx(r"osimertinib"): "osimertinib",
    rx(r"tagrisso"): "osimertinib",
 
    rx(r"crizotinib"): "crizotinib",
    rx(r"xalkori"): "crizotinib",
 
    rx(r"ceritinib"): "ceritinib",
    rx(r"zykadia"): "ceritinib",
 
    rx(r"alectinib"): "alectinib",
    rx(r"alecensa"): "alectinib",
 
    rx(r"brigatinib"): "brigatinib",
    rx(r"alunbrig"): "brigatinib",
 
    rx(r"alpelisib"): "alpelisib",
    rx(r"piqray"): "alpelisib",
 
    rx(r"copanlisib"): "copanlisib",
    rx(r"aliqopa"): "copanlisib",
 
    rx(r"duvelisib"): "duvelisib",
    rx(r"copiktra"): "duvelisib",
 
    rx(r"ibrutinib"): "ibrutinib",
    rx(r"imbruvica"): "ibrutinib",
 
    rx(r"acalabrutinib"): "acalabrutinib",
    rx(r"calquence"): "acalabrutinib",
 
    rx(r"zanubrutinib"): "zanubrutinib",
    rx(r"brukinsa"): "zanubrutinib",
    
    # Bisphosphonates
    rx(r"man"+SEP+r"lp@zol"): "zoledronic acid",
    rx(r"mannosylated"+SEP+r"liposome"+SEP+r"zol"): "zoledronic acid",
    rx(r"zoledronic"+SEP+r"acid"): "zoledronic acid",
    rx(r"zoledronate"): "zoledronic acid",
    rx(r"zometa"): "zoledronic acid",
    rx(r"zol"): "zoledronic acid",

    rx(r"alendronate"): "alendronate",
    rx(r"fosamax"): "alendronate",

    
    # Immunomodulators
    rx(r"imiquimod"): "imiquimod",
    rx(r"r837"): "imiquimod",

    rx(r"r848"): "R848",
    rx(r"resiquimod"): "R848",

    rx(r"cpg"+SEP+r"odn"): "CpG Oligodeoxynucleotide",
    rx(r"cpg"+SEP+r"oligodeoxynucleotide"): "CpG Oligodeoxynucleotide",

    rx(r"poly"+SEP+r"i:c"): "poly I:C",
    rx(r"polyinosinic"+SEP+r"polycytidylic"+SEP+r"acid"): "poly I:C",

    rx(r"sting"+SEP+r"agonist"): "STING agonist",
    rx(r"stimulator"+SEP+r"of"+SEP+r"interferon"+SEP+r"genes"): "STING agonist",

    rx(r"cgamp"): "cGAMP",
    rx(r"2'?3'?"+SEP+r"cgamp"): "cGAMP",

    rx(r"dmxaa"): "vadimezan",
    rx(r"vadimezan"): "vadimezan",

    rx(r"trail"): "dulanermin",             # TRAIL is the alias; canonical = dulanermin
    rx(r"tnf"+SEP+r"related"+SEP+r"apoptosis"+SEP+r"inducing"+SEP+r"ligand"): "dulanermin",
    rx(r"dulanermin"): "dulanermin",

    rx(r"tlr9"): "TLR9 agonist",
    rx(r"toll"+SEP+r"like"+SEP+r"receptor"+SEP+r"9"): "TLR9 agonist",

    rx(r"tnf"+SEP+r"[αa]"): "TNF-α",
    rx(r"tumor"+SEP+r"necrosis"+SEP+r"factor"+SEP+r"[αa]"): "TNF-α",
    rx(r"tnfalpha"): "TNF-α",
    rx(r"tnf"+SEP+r"alpha"): "TNF-α",

    
    # Antibodies / Peptides
    rx(r"trastuzumab"): "trastuzumab",
    rx(r"herceptin"): "trastuzumab",

    rx(r"anti"+SEP+r"pd"+SEP+r"l1"): "anti-PD-L1",
    rx(r"pd"+SEP+r"l1"+SEP+r"antibody"): "anti-PD-L1",

    rx(r"rgd"+SEP+r"peptide"): "RGD peptide",
    rx(r"arg"+SEP+r"gly"+SEP+r"asp"): "RGD peptide",

    rx(r"irgd"): "iRGD",

    rx(r"c"+SEP+r"peptide"+SEP+r"sln"+SEP+r"ptx"): "C-peptide-SLN-PTX",

    # Checkpoint inhibitors
    rx(r"pembrolizumab"): "keytruda",
    rx(r"keytruda"): "keytruda",
 
    rx(r"nivolumab"): "opdivo",
    rx(r"opdivo"): "opdivo",
 
    rx(r"atezolizumab"): "tecentriq",
    rx(r"tecentriq"): "tecentriq",
 
    rx(r"ipilimumab"): "yervoy",
    rx(r"yervoy"): "yervoy",
 
    rx(r"avelumab"): "avelumab",
    rx(r"bavencio"): "avelumab",
 
    rx(r"durvalumab"): "durvalumab",
    rx(r"imfinzi"): "durvalumab",  

    
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

    rx(r"ber"): "berberine",                # BER is the alias; canonical = berberine
    rx(r"berberine"): "berberine",

    rx(r"lupeol"): "lupeol",
    rx(r"fagarasterol"): "lupeol",
    rx(r"fagarsterol"): "lupeol",
    rx(r"monogynol B"): "lupeol",
    rx(r"clerodol"): "lupeol",
    rx(r"farganasterol"): "lupeol",
    rx(r"lupenol"): "lupeol",
    rx(r"tsl-lup\d+"): "lupeol",
 
    # Hematopoietic growth factors
    rx(r"gcsf"): "filgrastim",
    rx(r"g"+SEP+r"csf"): "filgrastim",
    rx(r"granulocyte"+SEP+r"colony"+SEP+r"stimulating"+SEP+r"factor"): "filgrastim",
    rx(r"endogenous"+SEP+r"g"+SEP+r"csf"): "filgrastim",
    rx(r"g"+SEP+r"csf"+SEP+r"signaling"): "filgrastim",
    rx(r"filgrastim"): "filgrastim",
    rx(r"neupogen"): "filgrastim",
 
    rx(r"pegfilgrastim"): "pegfilgrastim",
    rx(r"peg"+SEP+r"filgrastim"): "pegfilgrastim",
    rx(r"neulasta"): "pegfilgrastim",
 
    rx(r"lenograstim"): "lenograstim",
    rx(r"granocyte"): "lenograstim",
})

# 4. COMPILE
COMPILED_MAP = [
    (re.compile(pattern, re.I), canonical)
    for pattern, canonical in cargo_map.items()
]

# VOCABULARY REGISTRIES
# These are your keyword lists and regex patterns for direct extraction (no LLM needed)

# type
ORGANIC_NP_KEYWORDS = [
    "lipid-polymer","liposome", "lipid nanoparticle", "solid lipid nanoparticle", "liposomal",
    "polymeric nanoparticle", "micelle", "dendrimer", "niosome",
    "exosome", "polymersome", "nanoemulsion", "lipoplex", "polyplex",
    "nanostructured lipid carrier", "cationic liposome", "ionizable LNP", "small unilamellar vesicles", "multilamellar",
]

ORGANIC_NP_ABBR = ["NP", "LNP", "LNPs", "PLGA", "PLA", "SLN", "NLC", "SUV", "MLV", "MLVs", "LUV", "NLC"]

INORGANIC_NP_KEYWORDS = [
    "gold nanoparticle", "iron oxide", "silica", "quantum dot", "silver nanoparticle",
    "zinc oxide", "titanium dioxide", "carbon nanotube", "graphene",
    "mesoporous silica", "calcium phosphate", "copper sulfide",
    "manganese dioxide", "prussian blue",
]

INORGANIC_NP_ABBR = ["AuNP", "SPION", "MSN", "GdNP", "Fe3O4", "TiO2", "ZnO", "AgNP", "SiO2", "CNT", "QD", "SPION", "MSN"]
 
# subtype
SUBTYPE_MAP = {
    "liposome": {
        "keywords": ["liposome", "liposomal", "liposomes","unilamellar", "multilamellar"],
        "abbr": ["SUV", "MLV", "LUV"]
    },

    "LNP": {
        "keywords": ["lipid nanoparticle", "ionizable lipid nanoparticle"],
        "abbr": ["LNP", "LNPs"]
    },

    "SLN": {
        "keywords": ["solid lipid nanoparticle"],
        "abbr": ["SLN"]
    },

    "NLC": {
        "keywords": ["nanostructured lipid carrier"],
        "abbr": ["NLC"]
    },

    "polymeric": {
        "keywords": [ "polymeric nanoparticle", "polymersome", "polyplex",],
        "abbr": ["PLGA","PLA"]
    },

    "micelle": {
        "keywords": ["micelle", "polymeric micelle"],
        "abbr": []
    },

    "dendrimer": {
        "keywords": ["dendrimer",],
        "abbr": ["PAMAM"]
    },

    "niosome": {
        "keywords": ["niosome"],
        "abbr": ["NISM"]
    },

    "lipoplex": {
        "keywords": ["lipoplex", "cationic lipid-DNA complex"],
        "abbr": []
    }
}
 
# ── charge ────────────────────────────────────────────────────────────────────
POSITIVE_KEYWORDS = [
    "cationic", "positively charged", "positive charge", "positive zeta",
]
NEGATIVE_KEYWORDS = [
    "anionic", "negatively charged", "negative charge", "negative zeta",
]
NEUTRAL_KEYWORDS = [
    "neutral", "zwitterionic", "near-neutral", "PEGylated neutral",
]
IONIZABLE_KEYWORDS = [
    "ionizable", "ionizable lipid", "pH-ionizable", "pKa",
]
 
# ── shape ─────────────────────────────────────────────────────────────────────
SPHERE_KEYWORDS = ["spherical", "sphere", "spheroid"]
ROD_KEYWORDS    = ["rod", "rod-shaped", "elongated", "cylindrical"]
DISK_KEYWORDS   = ["disc", "disk", "discoidal", "flat nanoparticle"]
 
# ── lamellarity ───────────────────────────────────────────────────────────────
UNILAMELLAR_KEYWORDS   = [ "unilamellar", "small unilamellar", "large unilamellar",]
UNILAMELLAR_ABBR = ["SUV", "SUVs", "LUV"]

MULTILAMELLAR_KEYWORDS = ["multilamellar", "multi-lamellar"]
MULTILAMELLAR_ABBR = ["MLV", "MLVs"]
 
# ── lipid composition ─────────────────────────────────────────────────────────
LIPID_KEYWORDS = [
    # structural phospholipids
    "MSPC", "monostearoyl phosphatidylcholine", "SPC", "DPPC", "DSPC", "DOPC", "DOPE", "HSPC", "hydrogenated soy phosphatidylcholine", "DPPE", "DMPC", "POPC", "POPE",
    "phosphatidylcholine", "sphingomyelin", "span 60", "PS", 

    "span", "twen"
    # sterols
    "cholesterol hemisuccinate", "CHEMS", "cholesterol", "chol", "CHO", "choles-terol",
    # PEG-lipids
    "DSPE-PEG", "DSPE-PEG2000", "DSPE-MPEG2000", "DSPE- PEG2000", "DSPE-PEG_2000", "DSPE- PEG_2000", "DSPE- MPEG2000", "DSPE- MPEG" , "DSPE- PEG",
    "C14-PEG2000","C14- PEG2000", "PEG-DMG", "PEG- DMG", "PEG2000-Cer16",
    # ionizable lipids
    "DLin-MC3-DMA", "MC3", "SM-102", "ALC-0315",
    "lipid A9", "C12-200", "OF-02", "5A2-SC8", "CKK-E12", "DLin-KC2-DMA",
    "DODAP", "DOTAP", "DOTMA", "GenVoy-ILM", "Lipid H", "DLODAP",
    
    "Gd.DOTA.DSA", # Gd.DOTA.DSA for MRI imagining
    "CF750.DSA", # for lipid fluorescence imagining 
]


NORMALIZATION_MAP = {
    "monostearoyl phosphatidylcholine": "MSPC",
    "choles-terol":"cholesterol",
    "CHO": "cholesterol",
    "chol": "cholesterol",
    "DSPE- PEG2000": "DSPE-PEG2000",
    "DSPE- MPEG2000": "DSPE-MPEG2000",
    "DSPE- PEG_2000": "DSPE-PEG2000",
    "DSPE-PEG_2000": "DSPE-PEG2000",
    "PEG-DMG": "DMG-PEG2000",  # escolher forma dominante
    "hydrogenated soy phosphatidylcholine": "HSPC",
    "cholesterol hemisuccinate": "CHEMS",
    "sphingomyelin": "SM",
    "phosphatidylserine": "PS",
}

PHOSPHATIDYLCHOLINES = {"SPC","DPPC", "DSPC", "DOPC", "DOPE", "HSPC", "DPPE", "DMPC", "POPC", "POPE"}

GENERIC_TERMS = {
      "phosphatidylcholine": PHOSPHATIDYLCHOLINES,
}
 
# ── stimulus responsive ───────────────────────────────────────────────────────
STIMULUS_MAP = {
    "pH-sensitive":          ["pH-sensitive", "pH-responsive", "acid-responsive", "pH-triggered", "endosomal pH", "tumor acidic pH"],
    "Thermosensitive":       ["thermosensitive", "temperature-responsive", "heat-sensitive", "thermo-responsive", "LTSL"],
    "Redox-sensitive":       ["redox-sensitive", "redox-responsive", "GSH-responsive", "glutathione", "disulfide bond", "ROS-responsive", "reactive oxygen species", "H2O2-responsive"],
    "Enzyme-responsive":     ["enzyme-responsive", "MMP-responsive", "protease-triggered", "cathepsin", "furin-cleavable", "MMP-2", "MMP-9", "hyaluronidase-responsive"],
    "Light-triggered":       ["light-triggered", "photo-responsive", "photosensitive", "NIR-responsive", "azobenzene"],
    "Hypoxia-responsive":    ["hypoxia-responsive", "hypoxia-triggered", "nitroimidazole", "azobenzene hypoxia"],
}
 
# bioconjugation
COVALENT_KEYWORDS     = ["covalent", "conjugated", "conjugation", "crosslinked",
                          "thioether", "amide bond", "maleimide", "click chemistry",
                          "NHS ester", "EDC coupling", "disulfide conjugation", "mannose-modified" ]
NONCOVALENT_KEYWORDS  = ["noncovalent", "electrostatic", "adsorption",
                          "physical adsorption", "hydrophobic interaction"]
 
# PEG 
PEG_KEYWORDS     = ["PEGylated", "PEG-coated", "PEG-modified", "DSPE-PEG",
                     "DSPE-MPEG2000", "PEG2000", "stealth liposome", "DMG-PEG2000",
                     "C14-PEG2000", "long-circulating"  "DMG-PEG2000", "PEG-DMG","PEG2000-Cer16",]
NON_PEG_KEYWORDS = ["non-PEGylated", "without PEG", "PEG-free",
                     "PEG-alternative", "polysarcosine", "polyoxazoline", ]
 
# coating 
MONOLAYER_KEYWORDS  = ["monolayer", "single layer", "single coating"]
MULTILAYER_KEYWORDS = ["multilayer", "multi-layer", "layer-by-layer", "LbL",
                        "double coating", "lipid shell", "hybrid core-shell"]

SURFACE_MODIFIER_KEYWORDS = [
    "hyaluronic acid", "HA coating", "HA-coated", "HA-modified",
    "mannose-modified", "folate-modified", "transferrin-modified",
    "aptamer-functionalized", "antibody-conjugated", "decorated"
]
# targeting ─────────────────────────────────────────────────────────────────
ACTIVE_TARGETING_KEYWORDS = [
    "active targeting", "ligand-targeted", "receptor-targeted",
    "antibody-conjugated", "aptamer", "folate-targeted", "transferrin",
    "RGD", "peptide-decorated", "antibody-functionalized",
    # TNBC-relevant receptors
    "EGFR-targeted", "anti-EGFR", "HER1", "EGF receptor",
    "CD44-targeted", "hyaluronic acid-modified", "HA-coated",
    "TRAIL receptor", "DR5", "αvβ3 integrin",
    "PDL1-targeted", "anti-PD-L1 conjugated",
    "nucleolin-targeted", "AS1411",
]
 
# drug loading ──────────────────────────────────────────────────────────────
DRUG_LOADING_MAP = {
    "Passive entrapment": {
        "keywords": ["passive", "passive entrapment", "passive loading", "thin film hydration", "solvent injection"],
        "abbr": []
    },

    "Active loading (pH gradient)": {
        "keywords": ["active loading", "pH gradient", "remote loading", "ammonium sulfate", "citrate buffer"],
        "abbr": []
    },

    "Bilayer intercalation": {
        "keywords": ["bilayer intercalation", "membrane intercalation", "lipophilic", "hydrophobic drug"],
        "abbr": []
    },

    "Surface conjugation": {
        "keywords": ["surface conjugation", "surface conj", "surface-conjugated", "surface-adsorbed"],
        "abbr": []
    },

    "Nucleic acid complexation": {
        "keywords": ["complexation", "electrostatic complexation", "electrostatic encapsulation", "siRNA loading", "mRNA encapsulation", "plasmid condensation", "siRNA targeting", "mRNA"
        ],
        "abbr": ["N/P"]
    },

    "Antibody-drug conjugate (ADC)": {
        "keywords": [ "antibody-drug conjugate", "drug-linker", "site-specific conjugation"],
        "abbr": ["ADC", "DAR"]
    },

    "Proliposome method": {
        "keywords": [ "proliposome", "pro-liposome"],
        "abbr": []
    },

    "Solvent injection / Nanoprecipitation": {
        "keywords": ["ethanol injection", "solvent injection", "nanoprecipitation", "solvent displacement", "microfluidic mixing", "rapid mixing"],
        "abbr": []
    },

    "Reverse-phase evaporation": {
        "keywords": ["reverse-phase evaporation", "REV method"],
        "abbr": ["REV"]
    },

    "Hydration-assisted encapsulation": {
        "keywords": ["freeze-thaw", "freeze thaw", "rehydration", "hydration"],
        "abbr": []
    },

    "Size reduction / Homogenization": {
        "keywords": ["high-pressure homogenization", "extrusion", "sonication", "microfluidizer"],
        "abbr": ["HPH"]
    },

    "Co-loading": {
        "keywords": ["co-loaded", "dual-loaded", "co-encapsulation", "simultaneous loading"],
        "abbr": []
    }
}
 
# dosing 
SINGLE_DOSE_KEYWORDS   = ["single dose", "single injection", "one injection",
                           "single administration", "single i.v."]

NUMBER_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"

MULTIPLE_DOSE_PATTERNS = [
    rf"every\s+{NUMBER_WORDS}\s+days?",
    r"every\s+other\s+day",
    r"daily",
    r"weekly",
    r"biweekly",
    r"q\.?d\.?",
    r"b\.?i\.?d\.?",
    r"q\d+d",
    rf"\s+{NUMBER_WORDS}\s+times\s+a\s+week",
    r"repeated",
    r"multiple\s+dose",
    r"cycle",
    r"on days",
    r"days",
]
 
# route 
SYSTEMIC_KEYWORDS = ["intravenous", "intraperitoneal", "systemic", "tail vein","intraperitoneally"]
SYSTEMIC_ABBR = ["i.v.", "IV", "i.p.", "IP"]

LOCAL_KEYWORDS = ["intratumoral", "local", "direct injection",
                          "subcutaneous", "intraductal"]
LOCAL_ABBR = ["i.t.", "IT"]

IV_WORDS = ["intravenous", "intravenously", "tail vein",]
IV_ABBR = ["IV", "i.v."]

IV_EXCLUDED = ["iv breast cancer", "stage iv breast cancer"]


IT_WORDS = ["intratumoral", "intra-tumoral"]
IT_ABBR = ["IT", "i.t."]

IP_WORDS = ["intraperitoneal"]
IP_ABBR = ["IP", "i.p."]



# therapy types ─────────────────────────────────────────────────────────────
THERAPY_MAP = {
    "Chemotherapy":              {"keywords": ["chemotherapy", "chemo", "cytotoxic",
                                  "doxorubicin", "paclitaxel", "docetaxel",
                                  "gemcitabine", "carboplatin", "cisplatin",
                                  "eribulin", "capecitabine", "nab-paclitaxel",
                                  "abraxane", "sacituzumab govitecan"],
                                  "abbr": ["ADC"]},

    "Gene therapy":              {"keywords": ["gene therapy", "siRNA", "mRNA", "plasmid",
                                  "gene silencing", "gene delivery", "miRNA",
                                  "antisense oligonucleotide", "shRNA"], 
                                  "abbr": ["ASO","CRISPR", "cas9"]},   

    "Immunotherapy":             {"keywords": ["immunotherapy", "immune checkpoint",
                                  "PD-L1", "anti-PD1", "anti-PD-L1",
                                  "checkpoint inhibitor", "atezolizumab",
                                  "pembrolizumab", "nivolumab", "ipilimumab",
                                  "tumor microenvironment reprogramming",
                                  "macrophage polarization", "M1 polarization",
                                  "innate immune activation", "STING agonist",
                                  "toll-like receptor", "TLR agonist"], 
                                  "abbr": ["CTLA-4", "TIM-3", "LAG-3"]},

    "Photodynamic therapy":      {"keywords": ["photodynamic",  "photosensitizer",
                                  "ROS generation", "singlet oxygen",
                                  "chlorin e6"], 
                                  "abbr": ["PDT",]}, 

    "Photothermal therapy":      {"keywords": ["photothermal", "NIR irradiation",
                                  "laser irradiation", "indocyanine green",
                                  "gold nanorod", "copper sulfide"],
                                  "abbr": ["PTT"]},

    "Radiotherapy":              {"keywords": ["radiotherapy", "radiation therapy",
                                  "radiosensitization", "radiodynamic"], 
                                  "abbr": []},

    "Ultrasound":                {"keywords": ["ultrasound", "sonodynamic", "HIFU",
                                  "focused ultrasound"], 
                                  "abbr": ["HIFU"]},

    "Photoacoustic therapy":     {"keywords": ["photoacoustic therapy", "photoacoustic"], 
                                  "abbr": []},

    "Targeted therapy":          {"keywords": ["targeted therapy", "PARP inhibitor", "olaparib",
                                  "talazoparib", "CDK4/6 inhibitor", "palbociclib",
                                  "PI3K inhibitor", "mTOR inhibitor",
                                  "EGFR inhibitor", "erlotinib",
                                  "androgen receptor inhibitor", "bicalutamide"], 
                                  "abbr": []},

    "CAR-T / cell therapy":      {"keywords": ["CAR-T", "CAR T cell", "adoptive cell therapy",
                                  "NK cell therapy", "TIL therapy"], 
                                  "abbr": []},

    "Combination therapy":       {"keywords": ["combination therapy", "combined therapy", "synergistic", "co-delivery", "dual drug", "chemo-immunotherapy",
                                  "chemo-photothermal"],
                                  "abbr": []},
}
 
# study strategy ────────────────────────────────────────────────────────────
DIAGNOSIS_KEYWORDS    = ["diagnosis", "diagnostic", "imaging only",
                          "detection", "contrast agent"]
THERANOSTICS_KEYWORDS = ["theranostic", "theragnosis",
                          "combined imaging and therapy", "dual-function",
                          "image-guided therapy"]
 
# tumor model ───────────────────────────────────────────────────────────────
INVIVO_GENERIC_KEYWORDS = [
    "tumor volume", "tumor growth", "tumor regression",
    "PBS control", "treatment group", "body weight loss",
    "in vivo", "mouse model", "mice treated"
]

INVITRO_GENERIC_KEYWORDS = [
    "in vitro", "cell culture", "tissue culture",
    "primary cells", "cell line", "cultured cells",
    "ex vivo", "laboratory assay", "biological assay", 
    "experimental model",
]


XENOGRAFT_KEYWORDS    = ["xenograft", "human tumor", "human cell line",
                          "human cancer cells", "xenografts", "cell into", "cells in BALB/c nude mice", "in SCID mice" ] # MDA-MB-231-BrM brain metastasis model in SCID mice
ALLOGRAFT_KEYWORDS    = ["allograft", "syngeneic", "syngeneic model",
                          "murine tumor","4t1", "ct26", "b16", "llc", "e0771", "BALB/c mice"] # duvida em BALB/c mice
ORTHOTOPIC_KEYWORDS   = ["orthotopic"]
HETEROTOPIC_KEYWORDS  = ["heterotopic", "subcutaneous", "flank"]

PDX_KEYWORDS          = ["PDX", "patient-derived xenograft",
                          "patient-derived tumor"]
METASTATIC_MODEL_KEYWORDS = ["metastatic model", "lung metastasis",
                              "brain metastasis", "tail vein metastasis model",
                              "experimental metastasis"]
 
# immune status ─────────────────────────────────────────────────────────────
IMMUNOCOMPROMISED_KEYWORDS = [
    "nude mice", "athymic", "immunodeficient",
    "immunocompromised", "NCr nude",
]
IMMUNOCOMPROMISED_ABBR = [
     "SCID", "NOD/SCID", "NSG", "RAG",
]

IMMUNOCOMPETENT_KEYWORDS = [
    "BALB/c", "C57BL/6", "immunocompetent", "syngeneic", "intact immune",
    "FVB/N", "immune",
]
 
# cancer type ───────────────────────────────────────────────────────────────
CANCER_TYPE_MAP = {
    "Breast":   ["breast", "4T1", "MCF-7", "MDA-MB", "T47D", "BT-474", "SKBR3",
                  "ZR-75", "SUM149", "SUM159", "HCC1806", "HCC1937", "HCC70",
                  "MDA-MB-231", "MDA-MB-468", "MDA-MB-436", "Hs578T",
                  "BT549", "CAL-51", "TNBC"],
    "Lung":     ["lung", "A549", "H460", "LLC", "Lewis lung", "H1299",
                  "H1975", "PC9"],
    "Liver":    ["liver", "hepatocellular", "HCC", "HepG2", "Huh7"],
    "Brain":    ["brain", "glioma", "glioblastoma", "GBM", "U87", "U251",
                  "T98G", "LN229"],
    "Pancreas": ["pancreas", "pancreatic", "PANC-1", "BxPC-3", "MIA PaCa"],
    "Ovary":    ["ovary", "ovarian", "SKOV3", "A2780", "ID8", "OVCAR"],
    "Skin":     ["melanoma", "B16", "A375", "skin", "B16F10"],
    "Cervix":   ["cervix", "cervical", "HeLa"],
    "Colon":    ["colon", "colorectal", "HCT116", "SW480", "CT26", "HT-29"],
    "Prostate": ["prostate", "PC3", "LNCaP", "DU145", "22Rv1"],
}
 
# breast cancer subtype ─────────────────────────────────────────────────────
BREAST_SUBTYPE_MAP = {
    "Triple-Negative (TNBC)": ["triple-negative", "TNBC", "triple negative",
                                "MDA-MB-231", "MDA-MB-468", "MDA-MB-436",
                                "4T1", "SUM149", "SUM159", "HCC1806",
                                "HCC1937", "HCC70", "BT549", "CAL-51",
                                "Hs578T", "ER-negative PR-negative HER2-negative"],
    "TNBC BL1":               ["BL1", "basal-like 1", "BRCA1-mutated TNBC"],
    "TNBC BL2":               ["BL2", "basal-like 2"],
    "TNBC mesenchymal":       ["mesenchymal TNBC", "MSL", "mesenchymal stem-like"],
    "TNBC immunomodulatory":  ["immunomodulatory TNBC", "IM subtype",
                                "immune-rich TNBC", "TIL-high TNBC"],
    "TNBC LAR":               ["LAR", "luminal androgen receptor", "androgen receptor positive TNBC"],
    "HER2-enriched":          ["HER2-positive", "HER2+", "HER2-enriched",
                                "SKBR3", "BT-474"],
    "Luminal A":              ["Luminal A", "ER-positive", "MCF-7", "T47D"],
    "Luminal B":              ["Luminal B"],
    "metastatic":             ["metastatic", "metastasis", "metastatic TNBC",
                                "brain metastasis", "lung metastasis"],
}
 
# imaging modalities ────────────────────────────────────────────────────────
IMAGING_MAP = {
    "MRI": {
        "abbrs": ["MRI", "T1", "T2", "T1-weighted", "T2-weighted", "Gd.DOTA.DSA"],
        "keywords": ["magnetic resonance imaging", "relaxivity"]
    },

    "Fluorescence": {
        "abbrs": ["NIRF", "DiI", "DiD", "DiR", "FITC", "Cy5", "Cy7", "CF750.DSA"],
        "keywords": ["fluorescence", "fluorescent", "near-infrared fluorescence"]
    },

    "PET": {
        "abbrs": ["PET", "18F", "64Cu", "89Zr"],
        "keywords": ["positron emission"]
    },

    "CT": {
        "abbrs": ["CT", "µCT"],
        "keywords": ["computed tomography", "X-ray CT", "micro-CT"]
    },

    "Ultrasound": {
        "abbrs": [],
        "keywords": ["ultrasound imaging", "echography", "US imaging"]
    },

    "Photoacoustic": {
        "abbrs": ["PAI", "MSOT"],
        "keywords": ["photoacoustic", "optoacoustic"]
    },

    "Thermal": {
        "abbrs": [],
        "keywords": ["thermal imaging", "infrared thermography", "IR imaging"]
    },

    "Luminescence": {
        "abbrs": ["IVIS", "luciferase"],
        "keywords": ["luminescence", "bioluminescence", "chemiluminescence"]
    },

    "SPECT": {
        "abbrs": ["SPECT", "99mTc", "111In"],
        "keywords": ["single-photon emission"]
    },
    "Cytometry":{
        "abbrs": [],
        "keywords": ["cytometry"]
    },
}
 
# off-target organs note: hepatic accumulation = liver acc., 
ORGAN_KEYWORDS = [
    "hepatic", "splenic", "liver", "spleen", "kidney", "lung", "heart", "brain",
    "intestine", "stomach", "muscle", "bone marrow", "thymus", "bladder",
    "lymph node", "skin", "ovary", "thyroid gland",
    # Additional common organs / tissues
    "pancreas", "pancreatic",
    "adrenal", "adrenal gland",
    "prostate", "prostatic",
    "testis", "testes", "testicular",
    "uterus", "uterine",
    "cervix", "cervical",
    "vagina", "vaginal",
    "breast", "mammary",
    "esophagus", "esophageal",
    "colon", "colonic",
    "rectum", "rectal",
    "duodenum", "jejunum", "ileum",
    "gallbladder", "biliary", "bile duct",
    "appendix", "appendiceal",
    "tonsil", "tonsillar",
    "salivary gland", "parotid", "submandibular",
    "pituitary", "pituitary gland",
    "pineal",
    "hypothalamus",
    "spinal cord",
    "nerve", "neural",
    "blood", "vascular", "vessel",
    "aorta",
    "vein", "venous",
    "artery", "arterial",
    "peritoneum", "peritoneal",
    "pleura", "pleural",
    "pericardium", "pericardial",
    "diaphragm",
    "fat", "adipose",
    "soft tissue",
    "cartilage",
    "tendon",
    "ligament",
    "joint", "synovial",
    "eye", "ocular", "retina",
    "ear", "otic",
    "nose", "nasal",
    "sinus",
    "pharynx", "larynx", "trachea", "bronchus", "bronchial",
]

 
TUMOR_PATTERNS = r"""
(?:
    tumor\s*volume\s*(?:reduction|decrease|shrinkage)
  | (?:reduc(?:e|ed|tion|es)|decreas(?:e|ed|es)|shrink(?:age|ing|es)?)\s+(?:in\s+)?tumou?r\s*volume
  | tumou?r\s*volume.*?(?:reduc(?:e|ed|tion|es)|decreas(?:e|ed|es)|shrink(?:age|ing|es)?|inhibit(?:ed|ing|s)?)
  | tumou?r\s*growth\s*(?:inhibition|inhibited)
  | (?:inhibition|suppression)\s+of\s+tumou?r\s*growth
  | \bTGI\b
  | tumou?r.*?(?:shrinkage|regression)
  | reduc(?:e|ed|tion|es)
)
"""

EE_PATTERNS = r"""
(?:
    encapsulation\s*efficienc(?:y|ies)
  |efficienc(?:y|ies)\s+of\s+encapsulation
  |\bEE\b\s*%?
  |drug\s*loading\s*efficienc(?:y|ies)
  |loading\s*efficienc(?:y|ies)
  |efficienc(?:y|ies)\s+of\s+drug\s*loading
  |\bDL\b\s*%?
  |encapsulation\s*(?:rate|ratio)
  |(?:encapsulation|loading).{0,40}?(?:efficienc(?:y|ies))
  | (?:encapsulated).{0,40}?(?:efficienc(?:y|ies))
)
"""

DELIVERY_E_KEYWORDS = r"""
(?:delivery\s+efficiency(?:\W+\w+){0,12}?\W+reached
|delivery\s+efficiency(?:\W+\w+){0,10}?\W+was
|delivery\s+efficiency
|cellular\s+uptake
|uptake
|internalization
|transfection\s+efficiency
|transfection
|released)
"""

PART_SIZE = r"""
(?: 
particl(?:e|es)\s+size
|diamete(?:r|rs)
|size(?:s)?
)
"""

# HELPER FUNCTIONS
def _match_abbreviation(text: str, abbrs: List[str]) -> bool:
    for a in abbrs:
        pattern = r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])"
        if re.search(pattern, text):
            return True
    return False

def _first_keyword_match(text: str, keywords: List[str]) -> bool:
    """Return True if any keyword from the list is found in text (case-insensitive)."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def _abr_first_keyword_match(text: str, words: List[str], abbrs: List[str]) -> bool:
    return _first_keyword_match(text, words) or _match_abbreviation(text, abbrs)


def _all_keyword_matches(text: str, keywords: List[str]) -> List[str]:
    """Return all keywords from the list found in text (case-insensitive)."""
    text_lower = text.lower()
    return list({kw for kw in keywords if kw.lower() in text_lower})

def _map_exact_keywords(text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    """
    Given a dict of {label: [keywords]}, return the first label whose
    keywords are found in text. Returns None if no match.
    """
    for label, keywords in mapping.items():
        if _match_abbreviation(text, keywords):
            return label
    return None

def _map_keywords(text: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    """
    Given a dict of {label: [keywords]}, return the first label whose
    keywords are found in text. Returns None if no match.
    """
    for label, keywords in mapping.items():
        if _first_keyword_match(text, keywords):
            return label
    return None

def _is_excluded(text:str, excluded_phrases: List[str]) -> Optional[str]:
    text_lower = text.lower()
    return any(p in text_lower for p in excluded_phrases)


def _all_map_matches(text: str, mapping: Dict[str, List[str]]) -> List[str]:
    """Return all labels whose keywords are found in text."""
    found = []
    for label, data in mapping.items():
        if _abr_first_keyword_match(text, data["keywords"], data["abbr"]):
            found.append(label)
    return found

def _find_matches_with_positions(text:str, rules_dict: Dict[str, Dict[str, List[str]]]) -> List[Tuple[int, str]]:
    """
    rules_dict = {label: {"keywords": [...], "abbr": [...]}}
    returns: list of (position, label)
    """

    results = []
    text_lower = text.lower()

    for label, rules in rules_dict.items():

        # keywords
        for kw in rules["keywords"]:
            pos = text_lower.find(kw.lower())
            if pos != -1:
                results.append((pos, label))

        # abbreviations (regex for word boundary)
        for ab in rules["abbr"]:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(ab) + r"(?![A-Za-z0-9])"
            match = re.search(pattern, text)
            if match:
                results.append((match.start(), label))

    return results

def _extract_float_near_keyword(text: str, keyword_pattern: str) -> Optional[float]:
    """
    Extract the float whose position is closest to the keyword occurrence in the text.
    Closeness is measured by absolute character distance.
    """
    # Find keyword
    keyword_match = re.search(keyword_pattern, text, re.IGNORECASE)
    if not keyword_match:
        return None

    keyword_pos = keyword_match.start()

    # Find all numbers in text
    number_pattern = r"[-+]?\d*\.?\d+"
    candidates = [(m.start(), m.group()) for m in re.finditer(number_pattern, text)]

    if not candidates:
        return None

    # Pick number closest to keyword position
    closest_value = None
    min_distance = float("inf")

    for pos, num_str in candidates:
        distance = abs(pos - keyword_pos)
        if distance < min_distance:
            min_distance = distance
            try:
                closest_value = float(num_str)
            except ValueError:
                continue

    return closest_value


def _extract_value_unit_closest_to_keyword(
    text: str,
    keyword_pattern: str,
    units: list[str] = None,
    max_distance: int =400
) -> Optional[Tuple[float, Optional[str], Optional[float], Optional[str]]]:
    text = text.replace("−", "-")

    unit_pattern = "|".join(re.escape(u) for u in units)

    unit_matches = list(re.finditer(
        rf"(\d+(?:\.\d+)?)\s*({unit_pattern})",
        text,
        re.IGNORECASE
    ))

    if not unit_matches:
        return None

    kw_matches = list(re.finditer(
        keyword_pattern,
        text,
        re.IGNORECASE | re.VERBOSE
    ))
    print(kw_matches)

    if not kw_matches:
        return None

    best = None
    best_dist = float("inf")

    for kw in kw_matches:
        kw_center = (kw.start() + kw.end()) / 2

        for unit in unit_matches:
            unit_center = (unit.start() + unit.end()) / 2
            dist = abs(unit_center - kw_center)

            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best = unit

    if best:
        value = float(best.group(1))
        unit = best.group(2)
        return value, None, None, unit

    return None
    return None


class NanoparticleExtractor(BaseModel):
    """
    Hybrid extractor: uses keyword search + regex + parsers for ~42 features,
    """
    _errors: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    # MATERIAL PROPERTIES

    def _extract_type(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, ORGANIC_NP_KEYWORDS):
            return "organic"
        if _first_keyword_match(text, INORGANIC_NP_KEYWORDS):
            return "inorganic"
        return None

    def _extract_subtype(self, text: str) -> Optional[str]:
        for subtype, rules in SUBTYPE_MAP.items():
            if _abr_first_keyword_match(text, rules["keywords"], rules["abbr"]):
                return subtype
            return None

    def _extract_lamellarity(self, text: str) -> Optional[str]:
        if _abr_first_keyword_match(text, MULTILAMELLAR_KEYWORDS, MULTILAMELLAR_ABBR):
            return "Multilamellar (MLV)"
        if _abr_first_keyword_match(text, UNILAMELLAR_KEYWORDS, UNILAMELLAR_ABBR):
            return "Unilamellar (SUV/LUV)"
        return None
    
    def _extract_charge_group(self, text: str) -> Optional[str]:
        
        if _first_keyword_match(text, POSITIVE_KEYWORDS):
            return "positive"
        if _first_keyword_match(text, NEGATIVE_KEYWORDS):
            return "negative"
        if _first_keyword_match(text, NEUTRAL_KEYWORDS):
            return "neutral"

        return None
     
    def _extract_zeta_potential(self, text: str) -> Optional[bool]:
        normalized = (text.replace("±", "+/-").replace("−", "-").replace("–", "-"))
        match = re.search(r"(-?\d+(?:\.\d+)?(?:\s*\+/-\s*\d+(?:\.\d+)?)?)\s*mV", normalized, re.IGNORECASE)
        #print (match.group(0).replace("+/-","±") if match else None)
        return True if match else False
        
    def _extract_size(self, text: str) -> Optional[bool]:
        result = _extract_value_unit_closest_to_keyword(text, PART_SIZE, ["nm"])
        #result = _extract_value_near_keyword(text, PART_SIZE, ["nm"])
        #print (result)
        return True if result else False 

    def _extract_pdi(self, text: str) -> Optional[bool]:
        PDI = (_extract_float_near_keyword(text, r"PDI|polydispersity index|polydispersity"))
        #print (PDI)
        return True if PDI else False

    def _extract_shape(self, text: str) -> Optional[str]:
        if _match_abbreviation(text, ROD_KEYWORDS):
            return "Rod"
        if _match_abbreviation(text, SPHERE_KEYWORDS):
            return "Sphere"
        if _match_abbreviation(text, DISK_KEYWORDS):
            return "Disk"
        return None
    
    # SURFACE ENGINEERING

    def _extract_coating_number(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, MULTILAYER_KEYWORDS):
            return "multilayer"
        if _first_keyword_match(text, MONOLAYER_KEYWORDS):
            return "monolayer"
        return None
    
    def _extract_stimulus_responsive(self, text: str) -> Optional[str]:
        result = _map_keywords(text, STIMULUS_MAP)
        return result if result else None

    def _extract_bioconjugation(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, COVALENT_KEYWORDS):
            return "covalent"
        return None

    def _extract_peg_coat(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, NON_PEG_KEYWORDS):
            return "non-PEGylated"
        if _first_keyword_match(text, PEG_KEYWORDS):
            return "PEGylated"
        return None

    def _extract_targeting_type(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, ACTIVE_TARGETING_KEYWORDS):
            return "active"
        return None

    def _extract_lipid_composition(self, text: str) -> Optional[List[str]]:
        pattern = r"\b(" + "|".join(map(re.escape, LIPID_KEYWORDS)) + r")\b"
        matches = [m.group(0) for m in re.finditer(pattern, text, re.IGNORECASE)]

        normalized = []
        for m in matches:
            m.lower()
            if m in NORMALIZATION_MAP:
                normalized.append(NORMALIZATION_MAP[m])
            else:
                normalized.append(m)

        seen = dict.fromkeys(normalized)
        normalized = list(seen)


        for generic, specific_set in GENERIC_TERMS.items():
            if generic in normalized and any(spec in normalized for spec in specific_set):
                normalized.remove(generic)

        return normalized if normalized else None
    

    def _extract_lipid_ratio(self, text: str) -> Optional[Dict]:
        match = re.search(
            r'\b\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)+\b',
            text
        )

        if not match:
            return None

        ratio_text = match.group()

        values = [float(n) for n in re.split(r'\s*[:/]\s*', ratio_text)]

        composition = self._extract_lipid_composition(text)

        if not composition:
            return None

        if len(values) < len(composition):
            return None

        if len(values) > len(composition):
            print("[WARNING]: more ratio values than lipid components. "
                "Check the LIPID_KEYWORDS and NORMALIZATION_MAP for consistency.")

        total = sum(values)

        if total == 0:
            return None

        percentages = [(v / total) * 100 for v in values]

        return {
            "ratios": percentages
        }
    
    def _extract_lipid_ratio_units(self, text: str) -> Optional[bool]:
        if self._extract_lipid_ratio(text) == None or len(self._extract_lipid_ratio(text)["ratios"]) != len(self._extract_lipid_composition(text)):
            return False
        return True 

    # DOSING & THERAPY
    def _extract_drug_loading_method(self, text: str) -> Optional[str]:
        for subtype, rules in DRUG_LOADING_MAP.items():
            if _abr_first_keyword_match(text, rules["keywords"], rules["abbr"]):
                return subtype
            return None
    
    def _extract_therapies(self, text: str):
        """Returns (therapy_a, therapy_b, therapy_c, combined_grouped)."""
        all_therapies = _all_map_matches(text, THERAPY_MAP)
        therapy_a = all_therapies[0] if len(all_therapies) > 0 else None
        therapy_b = all_therapies[1] if len(all_therapies) > 1 else None
        therapy_c = all_therapies[2] if len(all_therapies) > 2 else None
        if therapy_b: # or _first_keyword_match(text, "combined therapy"):
            combined  = "Combination therapy" 
        elif therapy_a == "Combination therapy":
            combined  = "Combination therapy"
        elif therapy_a:
            combined = "Monotherapy" 
        else: 
            combined = None
        return therapy_a, therapy_b, therapy_c, combined

    def _extract_study_strategy(self, text: str) -> Optional[str]:
        if _first_keyword_match(text, THERANOSTICS_KEYWORDS):
            return "Theranostics"
        if _first_keyword_match(text, DIAGNOSIS_KEYWORDS):
            return "Diagnosis"
        return None
        

    def _extract_route(self, text: str) -> Optional[list[str]]:
        rules = {
            "Local": {
                "keywords": LOCAL_KEYWORDS,
                "abbr": LOCAL_ABBR
            },
            "Systemic": {
                "keywords": SYSTEMIC_KEYWORDS,
                "abbr": SYSTEMIC_ABBR
            }
        }


        matches = _find_matches_with_positions(text, rules)
        matches.sort(key=lambda x: x[0])

        filtered = [
            (pos, label)
            for (pos, label) in matches
            if not (
                label == "Systemic"
                and _is_excluded(text, IV_EXCLUDED)
            )
        ]
        return [label for _, label in filtered]

    def _extract_route_subtype(self, text: str) -> Optional[list[str]]:
        rules = {
            "Intravenous": {
                "keywords": IV_WORDS,
                "abbr": IV_ABBR
            },
            "Intratumoral": {
                "keywords": IT_WORDS,
                "abbr": IT_ABBR
            },
            "Intraperitoneal": {
                "keywords": IP_WORDS,
                "abbr": IP_ABBR
            }
        }

        matches = _find_matches_with_positions(text, rules)
        matches.sort(key=lambda x: x[0])

        filtered = [
            (pos, label)
            for (pos, label) in matches
            if not (
                label == "Intravenous"
                and _is_excluded(text, IV_EXCLUDED)
            )
        ]

        return [label for _, label in filtered]

    def _extract_dose(self, text: str) -> Optional[bool]:
        """
        ParametersParser extracts mg/kg values if you add 'mg/kg' to
        quantity_units in your synthesis_parsing_parameters.json config.
        Fallback: direct regex.
        """
        # Dose por peso corporal (mg/kg, µg/kg, ...)
        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mg|µg|μg|ug|g)\s*/\s*kg\b",
            text,
            re.IGNORECASE,
        ):
            return True

        # Massa simples, mas NÃO seguida de mL ou L
        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mg|µg|μg|ug|g|kg)\b(?!\s*(?:/|\b(?:mL|L)\b))",
            text,
            re.IGNORECASE,
        ):
            return True

        return False
        

    def _extract_dosing_schedule(self, text: str) -> Optional[str]:
        if self._extract_dose(text)==False:
            return None
        if _first_keyword_match(text, SINGLE_DOSE_KEYWORDS):
            return "Single Dose"
        for pattern in MULTIPLE_DOSE_PATTERNS:
            if re.search(pattern, text):
                return "Multiple-dose"
        return None


    # BIOLOGICAL CONTEXT

    def _extract_tumor_model(self, text: str) -> Optional[str]:
        is_xeno  = _first_keyword_match(text, XENOGRAFT_KEYWORDS)
        is_allo  = _first_keyword_match(text, ALLOGRAFT_KEYWORDS)
        is_ortho = _first_keyword_match(text, ORTHOTOPIC_KEYWORDS)
        is_heter = _first_keyword_match(text, HETEROTOPIC_KEYWORDS)
        base = "Xenograft" if is_xeno else ("Allograft" if is_allo else None)
        loc  = "orthotopic" if is_ortho else ("heterotopic" if is_heter else None)
        if base and loc:
            return f"{base} {loc}"
        elif base:
            return f"{base}"
        elif loc:
            return f"{loc}"
        elif _first_keyword_match(text, INVIVO_GENERIC_KEYWORDS):
                return "In vivo generic keyword"
        else: 
            if _first_keyword_match(text, INVITRO_GENERIC_KEYWORDS):
                return "In vitro generic keyword"

    def _extract_immune_status(self, text: str) -> Optional[str]:
        if _abr_first_keyword_match(text, IMMUNOCOMPROMISED_KEYWORDS, IMMUNOCOMPROMISED_ABBR):
            return "Immunocompromised"
        if _first_keyword_match(text, IMMUNOCOMPETENT_KEYWORDS):
            return "Immunocompetent"
        return None

    def _extract_cancer_type(self, text: str) -> Optional[str]:
        return _map_exact_keywords(text, CANCER_TYPE_MAP)

    def _extract_breast_subtype(self, text: str) -> Optional[str]:
        return _map_exact_keywords(text, BREAST_SUBTYPE_MAP)

    def _extract_imaging(self, text: str):
        found = []
        for modality, patterns in IMAGING_MAP.items():
            if _abr_first_keyword_match(text, patterns["keywords"], patterns["abbrs"]):
                found.append(modality)

        a = found[0] if len(found) > 0 else None
        b = found[1] if len(found) > 1 else None
        c = found[2] if len(found) > 2 else None

        return a, b, c


    # OUTCOMES

    def _extract_ic50(self, text: str) -> bool:
        return True if re.search(r"IC\s*50|IC₅₀|IC_50", text, re.IGNORECASE) else False

    def _extract_encapsulation_efficiency(self, text: str) ->  Optional[bool]:
        EE = _extract_value_unit_closest_to_keyword(
            text, EE_PATTERNS, ["%"], 200
        )
        #print (EE)
        return True if EE else False 

    
    def _extract_delivery_efficiency(self, text: str) -> Optional[bool]:
        DE = _extract_value_unit_closest_to_keyword(
            text,
            DELIVERY_E_KEYWORDS,
            ["%"])
        #print (DE)
        return True if DE else False

    def _extract_distribution_half_life(self, text: str) -> Optional[bool]:
        DHL= _extract_float_near_keyword(
            text, r"t½α|t1/2α|distribution half.life|alpha half.life"
        )
        #print (DHL)
        return True if DHL else False

    def _extract_circulation_half_life(self, text: str) -> Optional[float]:
        CHL = _extract_float_near_keyword(
            text, r"t½β|t1/2β|circulation half.life|blood circulation.*half|elimination half.life|t1/2|t½"
        )
        #print (CHL)
        return True if CHL else False
    
    def _extract_biodistribution(self, text: str)-> Optional[bool]:
        """
        Returns list of (organ, pct) tuples sorted by % value descending.
        Looks for patterns like 'liver accumulation was 15%' or '20% in spleen'.
        """
        for organ in ORGAN_KEYWORDS:
            # pattern: organ name near a percentage value
            pattern = rf"(?:{organ}).{{0,80}}?([\d\.]+)\s*%|" \
                      rf"([\d\.]+)\s*%.{{0,80}}?(?:{organ})"
            match = re.search(pattern, text, re.IGNORECASE)
            
            return True if match else False 

    def _extract_tumor_vol_reduction(self, text: str) -> Optional[bool]:
        TVL = _extract_value_unit_closest_to_keyword(
            text,
            TUMOR_PATTERNS,
            ["%"])
        #print (TVL)
        return True if TVL else False
    
    # CARGO

    def extract_cargos(self, text: str)-> Optional[List]:
        """Return all unique cargos in order found."""
        found = []
        for pat, name in COMPILED_MAP:
            if pat.search(text) and name not in found:
                found.append(name)
        return found
    
    def cargo_category(self, name: str) -> Optional[str]:
        for cat, vals in CARGO_DB.items():
            if name in vals:
                return cat
        return None

    def extract_cargo_categories(self, text: str) -> Optional[list]:
        cargos_list = self.extract_cargos(text)

        if cargos_list:
            categories = []

            for item in cargos_list:
                c = self.cargo_category(item)
                categories.append(c)

            return categories

        return None


    def extract(self, text: str) -> NanoparticleData:
        """ 
        Extract all nanoparticle features from a paper section text.

        Args:
            text:    Relevant paper section text (after paragraph_classifier filtering)
            use_llm: Set False to skip LLM calls (e.g. for testing / fast runs)

        Returns:
            NanoparticleData with all extractable fields populated
        """ 
        therapy_a, therapy_b, therapy_c, combined = self._extract_therapies(text)
        imaging_a, imaging_b, imaging_c = self._extract_imaging(text)


        data = NanoparticleData(
            # Material properties
            type                    = self._extract_type(text),
            subtype                 = self._extract_subtype(text),
            size_nm                 = self._extract_size(text),
            charge_group            = self._extract_charge_group(text),
            shape                   = self._extract_shape(text),
            lamellarity             = self._extract_lamellarity(text),
            zeta_potential_mv       = self._extract_zeta_potential(text),
            pdi                     = self._extract_pdi(text),

            # Surface engineering
            coating_number          = self._extract_coating_number(text),
            lipid_composition       = self._extract_lipid_composition(text),
            lipid_composition_ratio = self._extract_lipid_ratio(text),
            lipid_composition_ratio_units = self._extract_lipid_ratio_units(text),
            stimulus_responsive     = self._extract_stimulus_responsive(text),
            bioconjugation_nature   = self._extract_bioconjugation(text),
            peg_coat                = self._extract_peg_coat(text),
            targeting_type          = self._extract_targeting_type(text),

            # Dosing & therapy
            drug_loading_method     = self._extract_drug_loading_method(text),
            dose_group        = self._extract_dose(text),
            no_days_dosing_grouped  = self._extract_dosing_schedule(text),
            route                   = self._extract_route(text),
            route_subtype           = self._extract_route_subtype(text),
            therapy_a               = therapy_a,
            therapy_b               = therapy_b,
            therapy_c               = therapy_c,
            combined_therapy_grouped= combined,
            study_strategy          = self._extract_study_strategy(text),

            # Biological context
            tumor_model             = self._extract_tumor_model(text),
            immune_status           = self._extract_immune_status(text),
            cancer_type             = self._extract_cancer_type(text),
            breast_cancer_subtype   = self._extract_breast_subtype(text),
            imaging_a               = imaging_a,
            imaging_b               = imaging_b,
            imaging_c               = imaging_c,

            # Outcomes
            ic50                    = self._extract_ic50(text),
            encapsulation_efficiency_pct = self._extract_encapsulation_efficiency(text),
            distribution_half_life_h = self._extract_distribution_half_life(text),
            circulation_half_life_h = self._extract_circulation_half_life(text),
            tumor_vol_reduction_pct = self._extract_tumor_vol_reduction(text),
            biodistribution = self._extract_biodistribution(text),
            cargos = self.extract_cargos(text),
            categories_cargos = self. extract_cargo_categories(text),

        )

        return data

    def extract_from_txt(
        self,
        txt_path: str,
        separator: str = "\n\n",
        save_json: str = None,
    ):
        """
        Apply self.extract() to each paragraph in txt
        Return JSON-ready results only.
        Works whether extract() returns dataclass, dict, or object.
        """

        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in text.split(separator) if p.strip()]

        results = []

        for i, paragraph in enumerate(paragraphs, start=1):
            data = self.extract(paragraph)

            # convert result safely
            if is_dataclass(data):
                extracted = asdict(data)
            elif isinstance(data, dict):
                extracted = data
            else:
                extracted = vars(data)

            results.append({
                "paragraph_id": i,
                "paragraph": paragraph,
                "extracted": extracted
            })

        if save_json:
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        return results