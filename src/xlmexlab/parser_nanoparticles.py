#parser
import re
from collections import OrderedDict
from pydantic import BaseModel, PrivateAttr
import Levenshtein
from typing import List

CARGO_DB = OrderedDict({
    "alcohol_antagonist": [
        "Antabuse",
        "DS",
    ],

    "alkaloid": [
        "Vinblastine",
        "Vincristine",
        "Vinorelbine",
    ],

    "alkylating_agent": [
        "BCNU",
        "Busulfan",
        "Carmustine",
        "Cyclophosphamide",
        "Melphalan",
        "Temozolomide",
        "TMZ",
    ],

    "anthracycline": [
        "Adriamycin",
        "daunorubicin",
        "DOX",
        "Doxil",
        "epirubicin",
        "Hydroxydaunorubicin",
        "idarubicin",
    ],

    "anti_malarial": [
        "Chloroquine",
        "Hydroxychloroquine",
    ],

    "anti_tumourigenic_cytokine": [
        "IFN-γ",
        "IL-2",
    ],

    "anti_tumourigenic_cytokine_nucleic_acid": [
        "IL-12 mRNA",
        "IL-23 mRNA",
    ],

    "antibody": [
        "anti-PD-L1",
        "Atezolizumab",
        "Avelumab",
        "Bavencio",
        "Durvalumab",
        "Herceptin",
        "Imfinzi",
        "Ipilimumab",
        "Keytruda",
        "Nivolumab",
        "Opdivo",
        "Pembrolizumab",
        "Tecentriq",
        "Yervoy",
    ],

    "antibody_anthracycline": [
        "sacituzumab govitecan",
        "Trodelvy",
    ],

    "antifungal": [
        "Amphotericin B",
    ],

    "antigen": [
        "neoantigen mRNA",
        "OVA",
        "OVA peptide",
        "Tumor lysate",
    ],

    "antigen_nucleic_acid": [
        "HER2 antigen mRNA",
    ],

    "antihelmintic_ribosome_inactivating": [
        "Mebendazole",
    ],

    "antimetabolite": [
        "5-Fluorouracil",
        "5-FU",
        "Cytarabine",
        "GEM",
        "MTX",
        "Pemetrexed",
    ],

    "antiviral_nucleic_acid": [
        "Acyclovir",
    ],

    "beta_blocker": [
        "Propranolol",
    ],

    "biologic": [
        "DM1",
        "Filgrastim",
        "G-CSF",
        "Granocyte",
        "Lenograstim",
        "Neulasta",
        "Neupogen",
        "Pegfilgrastim",
    ],

    "biphosphonates": [
        "Fosamax",
        "Man-LP@ZOL",
        "Man-NP@ZOL",
        "Mannosylated Liposome ZOL",
        "ZOL",
        "Zoledronate",
        "Zometa",
    ],

    "boron_neutron_capture_therapy": [
        "BPA",
        "BSH",
    ],

    "cdk4_6_inhibitor_autophagy_inhibitor": [
        "Abemaciclib",
        "Ibrance",
        "Kisqali",
        "Palbociclib",
        "Ribociclib",
        "Verzenio",
    ],

    "enzyme": [
        "DNase",
        "glucose oxidase",
        "L-asparaginase",
    ],

    "enzyme_inhibitor": [
        "Lynparza",
        "Talzenna",
    ],

    "estrogen_receptor_modulator": [
        "Fulvestrant",
        "Tamoxifen",
    ],

    "gnrh_agonist": [
        "Goserelin",
        "Leuprolide",
        "Lupron",
        "Zoladex",
    ],

    "hypoxic_cytotoxin": [
        "AQ4N",
        "Tirapazamine",
    ],

    "immunoadjuvant": [
        "cGAMP",
        "CpG ODN",
        "DMXAA",
        "imiquimod",
        "poly I:C",
        "R837",
        "R848",
        "Resiquimod",
        "STING agonist",
        "TLR9 agonist",
        "vadimezan",
    ],

    "kinase_inhibitor": [
        "Acalabrutinib",
        "Afatinib",
        "Alecensa",
        "Alectinib",
        "Aliqopa",
        "Alpelisib",
        "Alunbrig",
        "Bicalutamide",
        "Brigatinib",
        "Brukinsa",
        "Calquence",
        "Casodex",
        "Ceritinib",
        "Copanlisib",
        "Copiktra",
        "Crizotinib",
        "Duvelisib",
        "Erlotinib",
        "Gilotrif",
        "Gleevec",
        "Ibrutinib",
        "Imatinib",
        "Imbruvica",
        "Lapatinib",
        "Nexavar",
        "Osimertinib",
        "Piqray",
        "Rap",
        "Sirolimus",
        "Sorafenib",
        "Sunitinib",
        "Sutent",
        "Tagrisso",
        "Tarceva",
        "Tykerb",
        "Xalkori",
        "Zanubrutinib",
        "Zykadia",
    ],

    "ligand": [
        "C-peptide-SLN-PTX",
        "iRGD",
        "MMP-responsive peptide",
        "RGD peptide",
    ],

    "metal_compound": [
        "Gd-DTPA",
    ],

    "natural_product": [
        "BER",
        "Berberine",
        "clerodol",
        "Curcumin",
        "diferuloylmethane",
        "EGCG",
        "epigallocatechin gallate",
        "fagarasterol",
        "fagarsterol",
        "farganasterol",
        "Ginsenoside",
        "lupenol",
        "Lupeol",
        "monogynol B",
        "Quercetin",
        "Resveratrol",
        "tsl-lup",
    ],

    "nitroxide_radical": [
        "4-amino-TEMPO",
        "TEMPO",
    ],

    "non_steroidal_anti_inflammatory": [
        "Aspirin",
        "Celecoxib",
        "Indomethacin",
    ],

    "nucleic_acid": [
        "AKT siRNA",
        "anti-EGFR siRNA",
        "base editor BRCA1",
        "BCL-2 siRNA",
        "Cas12a mRNA",
        "Cas9 + ESR1 sgRNA",
        "Cas9 + HER2 sgRNA",
        "Cas9 + PIK3CA sgRNA",
        "Cas9 mRNA",
        "circRNA",
        "crRNA",
        "EGFR siRNA",
        "EZH2 siRNA",
        "generic siRNA",
        "gp100 mRNA",
        "HER2 saRNA",
        "hTERT circRNA",
        "let-7",
        "miR-10b",
        "miR-145",
        "miR-155",
        "miR-182-3p",
        "miR-200",
        "miR-21",
        "miR-34a",
        "miR-373",
        "miRNA",
        "MMP-9 siRNA",
        "mRNA",
        "mRNA-4157",
        "mRNA-4359",
        "MUC1 mRNA",
        "MUC1 saRNA",
        "NY-ESO-1 saRNA",
        "OX40L mRNA",
        "p53 mRNA",
        "PI3K siRNA",
        "saRNA",
        "sgRNA",
        "shRNA",
        "siRNA",
        "STAT3 siRNA",
        "survivin siRNA",
        "TGF-β trap mRNA",
        "TRP2 mRNA",
        "Twist1 siRNA",
        "VEGF siRNA",
        "WT1 antigen mRNA",
    ],

    "peripheral_vasostimulant": [
        "Nitroglycerin",
        "Sildenafil",
    ],

    "photosensitizer": [
        "BPD",
        "Ce6",
        "ICG",
        "Phthalocyanine",
        "Porphyrin",
        "Verteporfin",
    ],

    "photosensitizer_enzyme": [
        "peroxidase",
    ],

    "platinum_prodrug": [
        "Carboplatin",
        "Cisplatin",
        "Nedaplatin",
        "Oxaliplatin",
    ],

    "proteasome_inhibitor": [
        "Bortezomib",
        "Carfilzomib",
        "Kyprolis",
        "Velcade",
    ],

    "proteinogenic_amino_acid": [
        "L-arginine",
        "L-glutamine",
    ],

    "purine_analog": [
        "Cladribine",
        "Clofarabine",
        "Fludarabine",
    ],

    "radioactive_element": [
        "131I",
        "177Lu",
        "64Cu",
        "89Zr",
        "90Y",
    ],

    "reducing_and_complexing_thiol": [
        "DTT",
        "Glutathione",
        "GSH",
        "N-acetylcysteine",
        "NAC",
    ],

    "ribonuclease": [
        "Onconase",
        "Ranpirnase",
    ],

    "rna_synthesis_inhibitor": [
        "Actinomycin D",
        "eribulin",
        "Halaven",
        "α-amanitin",
    ],

    "taxane": [
        "Abraxane",
        "Cabazitaxel",
        "nab-paclitaxel",
        "PTX",
        "Taxol",
        "Taxotere",
    ],

    "topoisomerase_inhibitor": [
        "Camptothecin",
        "CPT",
        "Etoposide",
        "Irinotecan",
        "Topotecan",
    ],

    "tumour_necrosis_factor": [
        "dulanermin",
        "hTRAIL",
        "TNF-α",
        "TRAIL",
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
rx(r"her2[-_\s/]+antigen[-_\s/]+mrna"): "HER2 antigen-encoding mRNA",
    rx(r"antabuse"): "Disulfiram",
    rx(r"ds"): "Disulfiram",
    rx(r"vinblastine"): "Vinblastine",
    rx(r"vincristine"): "Vincristine",
    rx(r"vinorelbine"): "Vinorelbine",
    rx(r"bcnu"): "Carmustine",
    rx(r"busulfan"): "Busulfan",
    rx(r"carmustine"): "Carmustine",
    rx(r"cyclophosphamide"): "Cyclophosphamide",
    rx(r"melphalan"): "Melphalan",
    rx(r"temozolomide"): "Temozolomide",
    rx(r"tmz"): "Temozolomide",
    rx(r"adriamycin"): "Doxorubicin",
    rx(r"daunorubicin"): "Daunorubicin",
    rx(r"dox"): "Doxorubicin",
    rx(r"doxil"): "Doxorubicin (liposomal formulation)",
    rx(r"epirubicin"): "Epirubicin",
    rx(r"hydroxydaunorubicin"): "Doxorubicin",
    rx(r"idarubicin"): "Idarubicin",
    rx(r"chloroquine"): "Chloroquine",
    rx(r"hydroxychloroquine"): "Hydroxychloroquine",
    rx(r"ifn[- ]?γ"): "Interferon gamma",
    rx(r"il[- ]?2"): "Interleukin-2",
    rx(r"il[- ]?12[-_\s/]+mrna"): "Interleukin-12 mRNA",
    rx(r"il[- ]?23[-_\s/]+mrna"): "Interleukin-23 mRNA",
    rx(r"anti[- ]?pd[- ]?l1"): "Anti-Programmed Death-Ligand 1 antibody",
    rx(r"atezolizumab"): "Atezolizumab",
    rx(r"avelumab"): "Avelumab",
    rx(r"bavencio"): "Avelumab",
    rx(r"durvalumab"): "Durvalumab",
    rx(r"herceptin"): "Trastuzumab",
    rx(r"imfinzi"): "Durvalumab",
    rx(r"ipilimumab"): "Ipilimumab",
    rx(r"keytruda"): "Pembrolizumab",
    rx(r"nivolumab"): "Nivolumab",
    rx(r"opdivo"): "Nivolumab",
    rx(r"pembrolizumab"): "Pembrolizumab",
    rx(r"tecentriq"): "Atezolizumab",
    rx(r"yervoy"): "Ipilimumab",
    rx(r"sacituzumab[-_\s/]+govitecan"): "Sacituzumab Govitecan",
    rx(r"trodelvy"): "Sacituzumab Govitecan",
    rx(r"amphotericin[-_\s/]+b"): "Amphotericin B",
    rx(r"neoantigen[-_\s/]+mrna"): "Neoantigen-encoding mRNA",
    rx(r"ova"): "Ovalbumin",
    rx(r"ova[-_\s/]+peptide"): "Ovalbumin peptide",
    rx(r"tumor[-_\s/]+lysate"): "Tumour cell lysate",
    rx(r"mebendazole"): "Mebendazole",
    rx(r"5[- ]?fluorouracil"): "Fluorouracil",
    rx(r"5[- ]?fu"): "Fluorouracil",
    rx(r"cytarabine"): "Cytarabine",
    rx(r"gem"): "Gemcitabine",
    rx(r"mtx"): "Methotrexate",
    rx(r"pemetrexed"): "Pemetrexed",
    rx(r"acyclovir"): "Acyclovir",
    rx(r"propranolol"): "Propranolol",
    rx(r"dm1"): "Mertansine (emtansine)",
    rx(r"filgrastim"): "Filgrastim",
    rx(r"g[- ]?csf"): "Granulocyte Colony-Stimulating Factor",
    rx(r"granocyte"): "Lenograstim",
    rx(r"lenograstim"): "Lenograstim",
    rx(r"neulasta"): "Pegfilgrastim",
    rx(r"neupogen"): "Filgrastim",
    rx(r"pegfilgrastim"): "Pegfilgrastim",
    rx(r"fosamax"): "Alendronate",
    rx(r"man[- ]?lp@zol"): "Mannosylated Liposome Zoledronic Acid",
    rx(r"man[- ]?np@zol"): "Mannosylated Nanoparticle Zoledronic Acid",
    rx(r"mannosylated[-_\s/]+liposome[-_\s/]+zol"): "Mannosylated Liposome Zoledronic Acid",
    rx(r"zol"): "Zoledronic Acid",
    rx(r"zoledronate"): "Zoledronic Acid",
    rx(r"zometa"): "Zoledronic Acid",
    rx(r"bpa"): "Boronophenylalanine",
    rx(r"bsh"): "Sodium borocaptate",
    rx(r"abemaciclib"): "Abemaciclib",
    rx(r"ibrance"): "Palbociclib",
    rx(r"kisqali"): "Ribociclib",
    rx(r"palbociclib"): "Palbociclib",
    rx(r"ribociclib"): "Ribociclib",
    rx(r"verzenio"): "Abemaciclib",
    rx(r"dnase"): "Deoxyribonuclease",
    rx(r"glucose[-_\s/]+oxidase"): "Glucose oxidase",
    rx(r"l[- ]?asparaginase"): "L-asparaginase",
    rx(r"lynparza"): "Olaparib",
    rx(r"talzenna"): "Talazoparib",
    rx(r"fulvestrant"): "Fulvestrant",
    rx(r"tamoxifen"): "Tamoxifen",
    rx(r"goserelin"): "Goserelin",
    rx(r"leuprolide"): "Leuprolide",
    rx(r"lupron"): "Leuprolide",
    rx(r"zoladex"): "Goserelin",
    rx(r"aq4n"): "Banoxantrone",
    rx(r"tirapazamine"): "Tirapazamine",
    rx(r"cgamp"): "Cyclic GMP-AMP",
    rx(r"cpg[-_\s/]+odn"): "CpG Oligodeoxynucleotide",
    rx(r"dmxaa"): "Vadimezan (5,6-Dimethylxanthenone-4-acetic acid)",
    rx(r"imiquimod"): "Imiquimod",
    rx(r"poly[-_\s/]+i:c"): "Polyinosinic-polycytidylic acid",
    rx(r"r837"): "Imiquimod",
    rx(r"r848"): "Resiquimod",
    rx(r"resiquimod"): "Resiquimod",
    rx(r"sting[-_\s/]+agonist"): "STING agonist (generic)",
    rx(r"tlr9[-_\s/]+agonist"): "TLR9 agonist (generic)",
    rx(r"vadimezan"): "Vadimezan (DMXAA)",
    rx(r"acalabrutinib"): "Acalabrutinib",
    rx(r"afatinib"): "Afatinib",
    rx(r"alecensa"): "Alectinib",
    rx(r"alectinib"): "Alectinib",
    rx(r"aliqopa"): "Copanlisib",
    rx(r"alpelisib"): "Alpelisib",
    rx(r"alunbrig"): "Brigatinib",
    rx(r"bicalutamide"): "Bicalutamide",
    rx(r"brigatinib"): "Brigatinib",
    rx(r"brukinsa"): "Zanubrutinib",
    rx(r"calquence"): "Acalabrutinib",
    rx(r"casodex"): "Bicalutamide",
    rx(r"ceritinib"): "Ceritinib",
    rx(r"copanlisib"): "Copanlisib",
    rx(r"copiktra"): "Duvelisib",
    rx(r"crizotinib"): "Crizotinib",
    rx(r"duvelisib"): "Duvelisib",
    rx(r"erlotinib"): "Erlotinib",
    rx(r"gilotrif"): "Afatinib",
    rx(r"gleevec"): "Imatinib",
    rx(r"ibrutinib"): "Ibrutinib",
    rx(r"imatinib"): "Imatinib",
    rx(r"imbruvica"): "Ibrutinib",
    rx(r"lapatinib"): "Lapatinib",
    rx(r"nexavar"): "Sorafenib",
    rx(r"osimertinib"): "Osimertinib",
    rx(r"piqray"): "Alpelisib",
    rx(r"rap"): "Rapamycin",
    rx(r"sirolimus"): "Rapamycin",
    rx(r"sorafenib"): "Sorafenib",
    rx(r"sunitinib"): "Sunitinib",
    rx(r"sutent"): "Sunitinib",
    rx(r"tagrisso"): "Osimertinib",
    rx(r"tarceva"): "Erlotinib",
    rx(r"tykerb"): "Lapatinib",
    rx(r"xalkori"): "Crizotinib",
    rx(r"zanubrutinib"): "Zanubrutinib",
    rx(r"zykadia"): "Ceritinib",
    rx(r"c[- ]?peptide[- ]?sln[- ]?ptx"): "C-peptide paclitaxel solid lipid nanoparticle",
    rx(r"irgd"): "Internalizing RGD peptide",
    rx(r"mmp[- ]?responsive[-_\s/]+peptide"): "MMP-responsive peptide",
    rx(r"rgd[-_\s/]+peptide"): "Arg-Gly-Asp peptide",
    rx(r"gd[- ]?dtpa"): "Gadolinium Diethylenetriamine Pentaacetic Acid",
    rx(r"ber"): "Berberine",
    rx(r"berberine"): "Berberine",
    rx(r"curcumin"): "Curcumin",
    rx(r"diferuloylmethane"): "Curcumin",
    rx(r"egcg"): "Epigallocatechin Gallate",
    rx(r"epigallocatechin[-_\s/]+gallate"): "Epigallocatechin Gallate",
    rx(r"ginsenoside"): "Ginsenoside",
    rx(r"quercetin"): "Quercetin",
    rx(r"resveratrol"): "Resveratrol",
    rx(r"4[- ]?amino[- ]?tempo"): "4-Amino-TEMPO",
    rx(r"tempo"): "(2,2,6,6-Tetramethylpiperidin-1-yl)oxyl",
    rx(r"aspirin"): "Acetylsalicylic acid",
    rx(r"celecoxib"): "Celecoxib",
    rx(r"indomethacin"): "Indomethacin",
    rx(r"akt[-_\s/]+sirna"): "AKT-targeting siRNA",
    rx(r"anti[- ]?egfr[-_\s/]+sirna"): "Anti-EGFR siRNA",
    rx(r"base[-_\s/]+editor[-_\s/]+brca1"): "Base editor targeting BRCA1",
    rx(r"bcl[- ]?2[-_\s/]+sirna"): "BCL-2-targeting siRNA",
    rx(r"cas12a[-_\s/]+mrna"): "Cas12a nuclease mRNA",
    rx(r"cas9[-_\s/]+\+[-_\s/]+esr1[-_\s/]+sgrna"): "Cas9 + ESR1-targeting sgRNA",
    rx(r"cas9[-_\s/]+\+[-_\s/]+her2[-_\s/]+sgrna"): "Cas9 + HER2-targeting sgRNA",
    rx(r"cas9[-_\s/]+\+[-_\s/]+pik3ca[-_\s/]+sgrna"): "Cas9 + PIK3CA-targeting sgRNA",
    rx(r"cas9[-_\s/]+mrna"): "Cas9 nuclease mRNA",
    rx(r"circrna"): "Circular RNA",
    rx(r"crrna"): "CRISPR RNA",
    rx(r"egfr[-_\s/]+sirna"): "EGFR-targeting siRNA",
    rx(r"ezh2[-_\s/]+sirna"): "EZH2-targeting siRNA",
    rx(r"generic[-_\s/]+sirna"): "Generic siRNA",
    rx(r"gp100[-_\s/]+mrna"): "Glycoprotein 100 mRNA",
    rx(r"her2[-_\s/]+sarna"): "HER2 self-amplifying RNA",
    rx(r"htert[-_\s/]+circrna"): "hTERT circular RNA",
    rx(r"let[- ]?7"): "Let-7 microRNA",
    rx(r"mir[- ]?10b"): "MicroRNA-10b",
    rx(r"mir[- ]?145"): "MicroRNA-145",
    rx(r"mir[- ]?155"): "MicroRNA-155",
    rx(r"mir[- ]?182[- ]?3p"): "MicroRNA-182-3p",
    rx(r"mir[- ]?200"): "MicroRNA-200",
    rx(r"mir[- ]?21"): "MicroRNA-21",
    rx(r"mir[- ]?34a"): "MicroRNA-34a",
    rx(r"mir[- ]?373"): "MicroRNA-373",
    rx(r"mirna"): "MicroRNA",
    rx(r"mmp[- ]?9[-_\s/]+sirna"): "MMP-9-targeting siRNA",
    rx(r"mrna"): "Messenger RNA",
    rx(r"mrna[- ]?4157"): "Personalized neoantigen mRNA (Moderna)",
    rx(r"mrna[- ]?4359"): "Immune checkpoint mRNA (Moderna)",
    rx(r"muc1[-_\s/]+mrna"): "Mucin-1 mRNA",
    rx(r"muc1[-_\s/]+sarna"): "Mucin-1 self-amplifying RNA",
    rx(r"ny[- ]?eso[- ]?1[-_\s/]+sarna"): "NY-ESO-1 self-amplifying RNA",
    rx(r"ox40l[-_\s/]+mrna"): "OX40 Ligand mRNA",
    rx(r"p53[-_\s/]+mrna"): "Tumour protein p53 mRNA",
    rx(r"pi3k[-_\s/]+sirna"): "PI3K-targeting siRNA",
    rx(r"sarna"): "Self-Amplifying RNA",
    rx(r"sgrna"): "Single Guide RNA",
    rx(r"shrna"): "Short Hairpin RNA",
    rx(r"sirna"): "Small Interfering RNA",
    rx(r"stat3[-_\s/]+sirna"): "STAT3-targeting siRNA",
    rx(r"survivin[-_\s/]+sirna"): "Survivin-targeting siRNA",
    rx(r"tgf[- ]?[βb][-_\s/]+trap[-_\s/]+mrna"): "TGF-β trap-encoding mRNA",
    rx(r"trp2[-_\s/]+mrna"): "Tyrosinase-related protein 2 mRNA",
    rx(r"twist1[-_\s/]+sirna"): "Twist1-targeting siRNA",
    rx(r"vegf[-_\s/]+sirna"): "VEGF-targeting siRNA",
    rx(r"wt1[-_\s/]+antigen[-_\s/]+mrna"): "Wilms Tumour 1 antigen mRNA",
    rx(r"nitroglycerin"): "Nitroglycerin",
    rx(r"sildenafil"): "Sildenafil",
    rx(r"bpd"): "Benzoporphyrin Derivative",
    rx(r"ce6"): "Chlorin e6",
    rx(r"icg"): "Indocyanine Green",
    rx(r"phthalocyanine"): "Phthalocyanine",
    rx(r"porphyrin"): "Porphyrin",
    rx(r"verteporfin"): "Benzoporphyrin Derivative (BPD)",
    rx(r"peroxidase"): "Peroxidase",
    rx(r"carboplatin"): "Carboplatin",
    rx(r"cisplatin"): "Cisplatin",
    rx(r"nedaplatin"): "Nedaplatin",
    rx(r"oxaliplatin"): "Oxaliplatin",
    rx(r"bortezomib"): "Bortezomib",
    rx(r"carfilzomib"): "Carfilzomib",
    rx(r"kyprolis"): "Carfilzomib",
    rx(r"velcade"): "Bortezomib",
    rx(r"l[- ]?arginine"): "L-arginine",
    rx(r"l[- ]?glutamine"): "L-glutamine",
    rx(r"cladribine"): "Cladribine",
    rx(r"clofarabine"): "Clofarabine",
    rx(r"fludarabine"): "Fludarabine",
    rx(r"131i"): "Iodine-131",
    rx(r"177lu"): "Lutetium-177",
    rx(r"64cu"): "Copper-64",
    rx(r"89zr"): "Zirconium-89",
    rx(r"90y"): "Yttrium-90",
    rx(r"dtt"): "Dithiothreitol",
    rx(r"glutathione"): "Glutathione",
    rx(r"gsh"): "Glutathione",
    rx(r"n[- ]?acetylcysteine"): "N-acetylcysteine",
    rx(r"nac"): "N-acetylcysteine",
    rx(r"onconase"): "Onconase",
    rx(r"ranpirnase"): "Ranpirnase",
    rx(r"actinomycin[-_\s/]+d"): "Actinomycin D",
    rx(r"eribulin"): "Eribulin mesylate",
    rx(r"halaven"): "Eribulin",
    rx(r"α[- ]?amanitin"): "Alpha-amanitin",
    rx(r"abraxane"): "Paclitaxel (albumin-bound)",
    rx(r"cabazitaxel"): "Cabazitaxel",
    rx(r"nab[- ]?paclitaxel"): "Paclitaxel (albumin-bound)",
    rx(r"ptx"): "Paclitaxel",
    rx(r"taxol"): "Paclitaxel",
    rx(r"taxotere"): "Docetaxel",
    rx(r"camptothecin"): "Camptothecin",
    rx(r"cpt"): "Camptothecin",
    rx(r"etoposide"): "Etoposide",
    rx(r"irinotecan"): "Irinotecan",
    rx(r"topotecan"): "Topotecan",
    rx(r"dulanermin"): "Recombinant human TRAIL",
    rx(r"tnf[- ]?α"): "Tumour Necrosis Factor alpha",
    rx(r"trail"): "Dulanermin (recombinant human TRAIL)",
    rx(r"htrail"): "Dulanermin (recombinant human TRAIL)",
    rx(r"lupeol"): "Lupeol",
    rx(r"fagarasterol"): "Lupeol",
    rx(r"fagarsterol"): "Lupeol",
    rx(r"monogynol[-_\s/]+b"): "Lupeol",
    rx(r"clerodol"): "Lupeol",
    rx(r"farganasterol"): "Lupeol",
    rx(r"lupenol"): "Lupeol",
    rx(r"tsl[- ]?lup"): "Lupeol",
})

# 4. COMPILE
COMPILED_MAP = [
    (re.compile(pattern, re.I), canonical)
    for pattern, canonical in cargo_map.items()
]


ORGAN_MAP = {
    # Bladder
    "bladder": "bladder",
    "urinary bladder": "bladder",

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

    #bone marrow
    "bone marrow": "bone marrow",

    #bone
    "bone": "bone",

    # Heart
    "heart": "heart",
    "cardiac": "heart",
    "myocardial": "heart",

    #Intestine
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

    #Kidney
    "kidney": "kidney",
    "renal": "kidney",
    "nephric": "kidney",

    # Heart
    "heart": "heart",
    "cardiac": "heart",
    "myocardial": "heart",
    
    #Liver
    "liver": "liver",
    "hepatic": "liver",
    "hepatocyte": "liver",
    "hepatocellular": "liver",

    # Lung
    "lung": "lung",
    "pulmonary": "lung",
    "bronchus": "lung",
    "bronchial": "lung",
    "alveolar": "lung",
    
    #Muscle
    "muscle": "muscle",
    "skeletal muscle": "muscle",

    # Spleen
    "spleen": "spleen",
    "splenic": "spleen",

    # Stomach
    "stomach": "stomach",
    "gastric": "stomach",

    #Thymus
    "thymus": "thymus",

    #Thyroid gland
    "thyroid gland": "thyroid gland",

#não aparecem no paper
    # Pancreas -> nao aparece
    "pancreas": "pancreas",
    "pancreatic": "pancreas",

    # Gallbladder / biliary
    "gallbladder": "gallbladder",
    "biliary": "gallbladder",
    "bile duct": "gallbladder",

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
    "fat": "adipose tissue",
    "adipose": "adipose tissue",
    "soft tissue": "soft tissue",
    "cartilage": "cartilage",
    "tendon": "tendon",
    "ligament": "ligament",
    "joint": "joint",
    "synovial": "joint",


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
            "parameter_name", "value", "unit", "drug_name", "size_type", 
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
    

    def parse_schedule_response(self, response: str) -> dict[str, str]:
        """
        Parses the schedule LLM response into a dict of {drug_name: schedule}.
        Expected format per line:  DrugName | single_dose
        """
        results = {}

        for raw_line in response.strip().splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) < 2:
                continue

            drug_name = parts[0]
            schedule  = parts[1] if parts[1] else "unknown"

            results[drug_name] = schedule

        return results


    def update_schedule(self, T_and_F_list: dict, schedule_raw: str) -> dict:
        """
        Parses the raw schedule LLM response and injects 'schedule'
        into each entry of dose_group, matched by drug_name.
        """
        if "dose_group" not in T_and_F_list or not T_and_F_list["dose_group"]:
            return T_and_F_list

        parsed = self.parse_schedule_response(schedule_raw)
        print(f"  [PARSER.update_schedule] Parsed schedule: {parsed}")

        for entry in T_and_F_list["dose_group"]:
            drug_name = entry.get("drug_name")
            entry["schedule"] = parsed.get(drug_name, "unknown")
            print(f"  [PARSER.update_schedule] {drug_name} → {entry['schedule']}")

        return T_and_F_list

    def parse_missing_lipids(self, response: str) -> list[str]:
        response = response.strip()
        if not response or response.lower() in ("none", "n/a", "-"):
            return []

        items = []
        for raw_line in response.splitlines():
            line = raw_line.strip(" -•\t")
            if not line or line.lower() == "none":
                continue
            items.append(line)
        return items

    @staticmethod
    def _norm_key(s: str) -> str:
        """Lowercase + collapse whitespace, for case/space-insensitive comparisons."""
        return " ".join(s.strip().lower().split())

    def update_lipid_composition(
        self,
        T_and_F_list: dict,
        missing_raw: str,
        normalization_map: dict,
        generic_terms: dict,
    ) -> dict:
        if "lipid_composition" not in T_and_F_list:
            return T_and_F_list

        existing = T_and_F_list.get("lipid_composition") or []
        missing_items = self.parse_missing_lipids(missing_raw)
        print(f"  [PARSER.update_lipid_composition] LLM reported missing: {missing_items}")

        # normalized lookup: norm_key -> canonical form already present
        existing_norm = {self._norm_key(e): e for e in existing}

        # normalized lookup for the normalization map (handles e.g. "Mc3" vs "MC3")
        norm_map_lookup = {self._norm_key(k): v for k, v in normalization_map.items()}

        normalized_new = []
        for item in missing_items:
            item_key = self._norm_key(item)

            # resolve to canonical form via normalization_map (case-insensitive)
            canonical = norm_map_lookup.get(item_key, item)
            canonical_key = self._norm_key(canonical)

            # skip if it already exists (case/space-insensitive) in regex result or in this batch
            if canonical_key in existing_norm:
                print(f"  [PARSER.update_lipid_composition] Skipping '{item}' -> already present as '{existing_norm[canonical_key]}'")
                continue

            already_added = any(self._norm_key(n) == canonical_key for n in normalized_new)
            if already_added:
                continue

            normalized_new.append(canonical)

        merged = list(existing) + normalized_new
        # final de-dup pass, case/space-insensitive, keeps first occurrence
        seen = {}
        deduped = []
        for item in merged:
            key = self._norm_key(item)
            if key not in seen:
                seen[key] = True
                deduped.append(item)
        merged = deduped

        # re-apply generic-term suppression (e.g. drop "phosphatidylcholine" if a specific PC is present)
        for generic, specific_set in generic_terms.items():
            generic_key = self._norm_key(generic)
            specific_keys = {self._norm_key(s) for s in specific_set}
            merged_keys = {self._norm_key(m) for m in merged}

            if generic_key in merged_keys and merged_keys & specific_keys:
                merged = [m for m in merged if self._norm_key(m) != generic_key]

        T_and_F_list["lipid_composition"] = merged if merged else None
        return T_and_F_list
    
    def parse_load_status_response(self, response: str) -> dict[tuple, str]:
        results = {}
        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            value, unit, status = parts[0], parts[1], parts[2]
            key = self._make_size_key(value, unit)
            results[key] = status if status in ("loaded", "unloaded") else None
        return results

    def update_load_status(self, T_and_F_list: dict, load_raw: str) -> list[dict]:
        size_entries = T_and_F_list.get("size_nm")
        if not size_entries:
            return size_entries

        parsed = self.parse_load_status_response(load_raw)

        for entry in size_entries:
            key = self._make_size_key(entry.get("value"), entry.get("unit"))
            entry["load"] = parsed.get(key)  # None if not matched/unknown

        return size_entries

    @staticmethod
    def _make_size_key(value, unit) -> tuple:
        """Normalize value/unit pairs so float vs string vs whitespace doesn't break matching."""
        try:
            v = round(float(str(value).strip()), 4)
        except (TypeError, ValueError):
            v = str(value).strip() if value is not None else None
        u = str(unit).strip().lower() if unit is not None else None
        return (v, u)
    
    def parse_lipid_ratio_units_response(self, response: str) -> str | None:
        response = self.strip_think_blocks(response)  # se já tiveres este helper, reaproveita

        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            param_name, value = parts[0], parts[1]
            if param_name != "lipid_composition_ratio_units":
                continue
            if value.lower() in ("not_extractable", "not extractable", "", "-", "none"):
                return None
            return value

        return None
    
    def parse_formulation_registry(self, response: str) -> dict[str, dict]:
        registry = {}
        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            code, drug, load = parts[0], parts[1], parts[2]
            registry[code.upper()] = {
                "drug_name": None if drug.lower() in ("none", "", "-") else drug,
                "load": load if load in ("loaded", "unloaded") else None,
            }
        return registry

    def resolve_formulation_code(self, text: str, registry: dict) -> dict | None:
        """Find a known formulation code mentioned in `text` and return its registry entry."""
        for code, info in registry.items():
            if re.search(r"\b" + re.escape(code) + r"\b", text, re.IGNORECASE):
                return {"formulation_code": code, **info}
        return None
    
    def parse_cargo_category_check(self, response: str) -> dict[str, dict]:
        response = self.strip_think_blocks(response)
        results = {}
        for raw_line in response.strip().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            name, is_drug, category = parts[0], parts[1], parts[2]
            results[name] = {
                "is_drug": is_drug.lower() == "yes",
                "category": category,
            }
        return results