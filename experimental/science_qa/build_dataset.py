import json
from os import mkdir
from os.path import join, exists
from typing import List, Dict

import pandas as pd
import requests
from data_processing.qa_data import WordCounts
from tqdm import tqdm

from config import CORPUS_DIR, NDMC_TRAIN3
from data_processing.preprocessed_corpus import Preprocessor, DatasetBuilder
from data_processing.text_utils import get_paragraph_tokenizer
from dataset import Dataset
from experimental.mc_encoder import McQuestion, McDataset
from utils import flatten_iterable


class ElasticSearchParagraph(object):
    def __init__(self, text: List[List[str]], searchScore: float, index: str):
        self.text = text
        self.searchScore = searchScore
        self.index = index

    def to_json(self):
        return self.__dict__


class McAnswerOption(object):
    def __init__(self, answer: List[str], focus: str, assertion: str):
        self.answer = answer
        self.focus = focus
        self.assertion = assertion

    def to_json(self):
        return self.__dict__

    def __repr__(self) -> str:
        return str(self.answer)


class AristoMcQuestion(object):
    def __init__(self, question_id: str, question: List[str], raw_question: str, query: Dict,
                 answer_options: List[McAnswerOption], paragraphs: List[ElasticSearchParagraph], answer: int):
        self.question_id = question_id
        self.query = query
        self.question = question
        self.raw_question = raw_question
        self.answer_options = answer_options
        self.paragraphs = paragraphs
        self.answer = answer

    def to_json(self):
        json = dict(self.__dict__)
        json["answer_options"] = [x.to_json() for x in self.answer_options]
        json["paragraphs"] = [x.to_json() for x in self.paragraphs]
        return json

    @classmethod
    def from_json(cls, json):
        json = dict(json)
        if "query" not in json:
            json["query"] = None
        json["answer_options"] = [McAnswerOption(x["answer"], x["focus"], x["assertion"]) for x in json["answer_options"]]
        json["paragraphs"] = [ElasticSearchParagraph(x["text"], x["searchScore"], x["index"]) for x in
                                  json["paragraphs"]]
        return AristoMcQuestion(**json)


class AristoMcCorpus(object):
    TRAIN_FILE = "train.json"
    DEV_FILE = "dev.json"
    VOC_FILE = "voc.json"

    @classmethod
    def build(cls, train: List[AristoMcQuestion], dev: List[AristoMcQuestion]):
        base_dir = join(CORPUS_DIR, "aristo-mc")
        if not exists(base_dir):
            mkdir(base_dir)
        with open(join(base_dir, cls.TRAIN_FILE), "w") as f:
            json.dump([x.to_json() for x in train], f)
        with open(join(base_dir, cls.DEV_FILE), "w") as f:
            json.dump([x.to_json() for x in dev], f)
        voc = set()
        for q in (train + dev):
            voc.update(q.question)
            for option in q.answer_options:
                voc.update(option.answer)
            for para in q.paragraphs:
                for sent in para.text:
                    voc.update(sent)
        with open(join(base_dir, cls.VOC_FILE), "w") as f:
            f.write(" ".join(sorted(voc)))

    def __init__(self):
        base_dir = join(CORPUS_DIR, "aristo-mc")
        self.dir = base_dir

    def get_train(self):
        with open(join(self.dir, self.TRAIN_FILE), "r") as f:
            data = json.load(f)
        return [AristoMcQuestion.from_json(x) for x in data]

    def get_dev(self):
        with open(join(self.dir, self.DEV_FILE), "r") as f:
            data = json.load(f)
        return [AristoMcQuestion.from_json(x) for x in data]


class ExtractQueryParagraph(Preprocessor):
    def __init__(self, n_tokens: int):
        self.max_tokens = n_tokens

    def preprocess(self, question: List[AristoMcQuestion], _) -> List[McQuestion]:
        out = []
        for q in question:
            n_tokens = 0
            text = []
            for para in q.paragraphs:
                for sent in para:
                    if len(sent) + n_tokens < self.max_tokens:
                        text.append(sent)
                        n_tokens += len(sent)
                    elif n_tokens != self.max_tokens:
                        text.append(sent[:self.max_tokens-n_tokens])
            out.append(McQuestion(q.question_id, q.question, [x.answer for x in q.answer_options],
                                  text, q.answer))
        return out


class BuildMcTrainingData(DatasetBuilder):
    def __init__(self, train_batcher, eval_batcher):
        self.train_batcher = train_batcher
        self.eval_batcher = eval_batcher

    def build_stats(self, data) -> object:
        wc = Counter()
        for point in data:
            wc.update(point.question)
            for sent in point.context:
                wc.update(sent)
            for ans in point.answer_options:
                wc.update(ans)
        return WordCounts(wc)

    def build_dataset(self, data, evidence, is_train: bool) -> Dataset:
        return McDataset(data, self.train_batcher if is_train else self.eval_batcher)


def build_question(tokenizer, max_to_keep, cache, question_id, question, answer_text):
    question_id = str(question_id)
    cache_file = join(cache, question_id + ".json")
    word_tokenize, sent_tokenize = get_paragraph_tokenizer(tokenizer)

    if exists(join(cache, cache_file)):
        with open(cache_file, "r") as f:
            data = json.load(f)
        decomposed, query, hits = data
    else:
        response = requests.get("http://aristo-docker.dev.ai2:8087/decompose", {"text": question})
        if response.status_code != 200:
            raise ValueError()
        decomposed = response.json()["question"]
        response.close()

        query = {"searchQuery": decomposed, "maxResults": max_to_keep, "sourceTypes": ["elasticsearch"]}
        response = requests.post("http://aristo-background-knowledge.dev.ai2:8091/search", json=query)
        if response.status_code != 200:
            raise ValueError()

        hits = response.json()
        response.close()

        with open(cache_file, "w") as f:
            json.dump([decomposed, query, hits], f)

    selections = decomposed["selections"]
    if len(selections) <= 1:
        raise ValueError()

    answer_options = [McAnswerOption(x["answer"], x["focus"], x["assertion"]) for x in decomposed["selections"]]
    answer = [i for i, s in enumerate(decomposed["selections"]) if s["key"] == answer_text]
    if len(answer) != 1:
        raise ValueError()
    answer = answer[0]
    question_tokens = word_tokenize(decomposed["text"])

    hits = hits["resultsPerSourceType"]["elasticsearch"]
    out = []
    for paragraph in hits:
        text = [word_tokenize(x) for x in sent_tokenize(paragraph["searchHit"])]
        out.append(ElasticSearchParagraph(text, paragraph["searchScore"], paragraph["index"]))

    return AristoMcQuestion(question_id, question_tokens, question, query, answer_options, out, answer)


def build_da_question(tokenizer, max_to_keep, cache, question_id, question, answer_text):
    print("!!")
    question_id = str(question_id)
    cache_file = join(cache, question_id + ".json")
    word_tokenize, sent_tokenize = get_paragraph_tokenizer(tokenizer)

    # if exists(join(cache, cache_file)):
    #     with open(cache_file, "r") as f:
    #         data = json.load(f)
    #     decomposed, query, hits = data
    # else:
    print(question)
    query = {"searchQuery": {"text": question}, "maxResults": max_to_keep, "sourceTypes": ["elasticsearch"]}
    response = requests.post("http://aristo-background-knowledge.dev.ai2:8091/search", json=query)
    if response.status_code != 200:
        raise ValueError()

        # hits = response.json()
    # response.close()

        # with open(cache_file, "w") as f:
        #     json.dump([query, hits], f)
    return


def _build_question_tuple(x):
    return build_question(*x)


def build_mc_data(source, max_to_keep=50, tokenizer: str= "NLTK_AND_CLEAN", n_threads=1):
    cache = "/tmp/mc-cache"

    data = pd.read_csv(source)
    if not data["isMultipleChoice"].all():
        raise ValueError()
    if data["hasDiagram"].any():
        raise ValueError()

    points = list(data[["id", "questionText", "answerText"]].itertuples(index=False, name=None))
    if n_threads == 1:
        return [build_question(tokenizer, max_to_keep, cache, *x) for x in tqdm(points)]
    else:
        from multiprocessing.pool import ThreadPool
        pbar = tqdm(total=len(points))
        out = []
        with ThreadPool(n_threads) as pool:
            for r in pool.imap_unordered(_build_question_tuple, [([tokenizer, max_to_keep, cache]+list(x)) for x in points]):
                pbar.update(1)
                out.append(r)
        pbar.close()
        return out


def _build_da_question_tuple(x):
    return build_da_question(*x)


def build_da_data(source, max_to_keep=50, tokenizer: str= "NLTK_AND_CLEAN", n_threads=1):
    cache = "/tmp/mc-cache"

    data = pd.read_csv(source)
    if data["isMultipleChoice"].any():
        raise ValueError()
    if data["hasDiagram"].any():
        raise ValueError()

    points = list(data[["id", "questionText", "answerText"]].itertuples(index=False, name=None))
    if n_threads == 1:
        return [build_da_question(tokenizer, max_to_keep, cache, *x) for x in tqdm(points)]
    else:
        from multiprocessing.pool import ThreadPool
        pbar = tqdm(total=len(points))
        out = []
        with ThreadPool(n_threads) as pool:
            for r in pool.imap_unordered(_build_da_question_tuple, [([tokenizer, max_to_keep, cache]+list(x)) for x in points]):
                pbar.update(1)
                out.append(r)
        return out


def build_mc_corpus(max_to_keep=50, tokenizer: str="NLTK_AND_CLEAN"):
    # print("Buliding dev...")
    # dev = build_mc_data(NDMC_DEV, max_to_keep, tokenizer, n_threads=1)

    print("Buliding train...")
    train = build_mc_data(NDMC_TRAIN3, 1)
    # train = None

    # print("Saving..")
    # AristoMcCorpus.build(train, dev)
    # print("Done!")
    # corp = AristoMcCorpus()
    # corp.get_train()
    # corp.get_dev()


def check():
    data = []
    for x in [NDMC_TRAIN3]:
        data.append(pd.read_csv(x))
    print(len(set(flatten_iterable(x.id for x in data))))

    cache = "/tmp/mc-cache"
    missing = 0
    for q in flatten_iterable(list(x.id) for x in data):
        q = str(q)
        cache_file = join(cache, q + ".json")
        if not exists(join(cache, cache_file)):
            missing += 1

    print(missing)


def build_da_corpus(max_to_keep=50, tokenizer: str="NLTK_AND_CLEAN"):
    print("Buliding dev...")
    # dev = build_da_data(NDDA_DEV, max_to_keep, tokenizer, n_threads=1)

    print("Buliding train...")
    train = build_mc_data(NDMC_TRAIN3)
    train = None
    #
    # print("Saving..")
    # AristoMcCorpus.build(train, dev)
    # print("Done!")
    # corp = AristoMcCorpus()
    # corp.get_train()
    # corp.get_dev()


if __name__ == "__main__":
    # check()
    build_mc_corpus()