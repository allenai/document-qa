from os.path import join, expanduser, dirname

"""
Global config options, its admittedly a bit hacky to have this be a python file,
also very convenient for IDEs
"""

ELASTIC_SEARCH_HOST = "aristo-es1.dev.ai2:9200"
VEC_DIR = join(expanduser("~"), "data", "glove")
WIKI_EXTRACT_DIR = join(expanduser("~"), "data", "wikipedia/extracted")
SQUAD_SOURCE_DIR = join(expanduser("~"), "data", "squad")
SQUAD_TRAIN = join(SQUAD_SOURCE_DIR, "train-v1.1.json")
SQUAD_DEV = join(SQUAD_SOURCE_DIR, "dev-v1.1.json")


ARISTO_SOURCE_DIR = join(expanduser("~"), "data", "aristo")
NDMC_DEV = join(ARISTO_SOURCE_DIR, "Omnibus-Gr08-NDMC-Dev.csv")
NDMC_TRAIN1 = join(ARISTO_SOURCE_DIR, "Omnibus-Gr08-NDMC-Train1.csv")
NDMC_TRAIN2 = join(ARISTO_SOURCE_DIR, "Omnibus-Gr08-NDMC-Train2.csv")
NDMC_TRAIN3 = join(ARISTO_SOURCE_DIR, "Omnibus-Gr08-NDMC-Train3.csv")
NDDA_DEV = join(ARISTO_SOURCE_DIR, "Omnibus-Gr08-NDDA-Dev.csv")
NDDA_TRAIN = join(ARISTO_SOURCE_DIR, "Omnibus-Gr08-NDDA-Train.csv")


REPHRASE_SERVICE = "http://aristo-docker.dev.ai2:8087/decompose"

TRIVIA_QA = join(expanduser("~"), "data", "trivia-qa")
TRIVIA_QA_UNFILTERED = join(expanduser("~"), "data", "triviaqa-unfiltered")

NIKET_QA = join(expanduser("~"), "data", "niket", "v2")


_BASE = dirname(__file__)
CORPUS_DIR = join(_BASE, "data")
