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
        "paclitaxel", "docetaxel", "doxorubicin",
        "epirubicin", "gemcitabine", "fluorouracil",
        "eribulin", "mertansine",
        "sacituzumab govitecan"                    
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
        "EGCG", "berberine" 
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

DEFAULT_FIELDS = ["parameter_name", "value", "unit", "condition"]

PARAM_META = {

    "size_nm": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "zeta_potential_mv": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "pdi": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "encapsulation_efficiency_pct": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "ic50": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "distribution_half_life_h": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "circulation_half_life_h": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "dose_group": {
        "fields": [
            "parameter_name", "value", "unit", "drug_name", "schedule",
        ],
    },

    "tumor_vol_reduction_pct": {
        "fields": [
            "parameter_name", "value", "unit", "drug_name",
        ],
    },

    "delivery_efficiency": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "biodistribution": {
        "fields": [
            "parameter_name", "value", "unit", "condition"
        ],
    },

    "lipid_composition_ratio_units": {
        "fields": [
            "parameter_name", "dimension"
        ]
    },
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
        
    def normalize_value(self, field: str, value):

        if value in ("", "-", "none", "not_extractable", None):
            return None

        value = value.strip()

        # convert numeric values automatically
        if field == "value":
            try:
                return float(value)
            except Exception:
                return value

        return value
    
    def parse_response(self, response: str) -> dict[str, list[dict]]:

        results = {}

        for raw_line in response.strip().splitlines():

            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) < 1:
                continue

            raw_param = parts[0]

            param = self.correct_param(
                raw_param,
                self._parameters
            )

            meta = PARAM_META.get(param, {})

            fields = meta.get("fields", DEFAULT_FIELDS)

            # remove parameter_name
            data_fields = fields[1:]

            entry = {}

            for idx, field in enumerate(data_fields, start=1):

                value = parts[idx] if idx < len(parts) else None

                value = self.normalize_value(field, value)

                entry[field] = value

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
    
    def postprocess_ratio(self, lipids_list: list[str], lipids_ratio_list: list[str], dimension: str) -> dict:
        if lipids_list is not None and lipids_ratio_list is not None:
            if len(lipids_list) != len(lipids_ratio_list):
                lipids_list = None
                lipids_ratio_list = None
            else: 
                    processed = []
                    for lipid, ratio in zip(lipids_list, lipids_ratio_list):
                        processed.append({
                            "lipid": lipid,
                            "ratio": ratio,
                            "distribution": dimension,
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
                "drug_name": item.get("drug_name"),
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

    def replace(self, T_and_F_list: dict, simulated_llm_response: str) -> dict:

        parsed = self.parse_response(simulated_llm_response)

        postprocessors = {
            "biodistribution":
                self.postprocess_biodistribution,

            "ic50":
                self.postprocess_IC50,

            "dose_group":
                self.postprocess_dose_group,
        }

        for key, fn in postprocessors.items():

            if key in parsed:
                parsed[key] = fn(parsed[key])

        for key, value in parsed.items():

            if key in T_and_F_list:
                T_and_F_list[key] = value

        if (
            "zeta_potential_mv" in T_and_F_list
            and not T_and_F_list.get("charge_group")
        ):

            charge = self.compute_charge_group(
                T_and_F_list["zeta_potential_mv"]
            )

            T_and_F_list["charge_group"] = charge

        return T_and_F_list


