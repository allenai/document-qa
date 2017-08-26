import json
from os import listdir
import numpy as np
from os.path import join

import requests

from science_qa.build_dataset import AristoMcCorpus
from utils import flatten_iterable


def main():
    corp = AristoMcCorpus()
    questions = corp.get_train()

    questions = []
    for filename in sorted(listdir("/tmp/mc-cache/")):
        with open(join("/tmp/mc-cache/", filename), "r") as f:
            decomposed, query, hits = json.load(f)
        print(decomposed["rawQuestion"])
        print("\n".join([x["searchHit"] for x in hits["resultsPerSourceType"]["elasticsearch"]]))
        print()

    # questions.sort(key=lambda x: x.question_id)
    # np.random.RandomState(0).shuffle(questions)

    # for ix, q in enumerate(questions):
        # print(" ".join(q.question))
        # text = flatten_iterable(flatten_iterable(para.text) for para in q.paragraphs)
        # print(len(text))
        # print(q.answer_options[q.answer])
        # print(sum(sum(len(s) for s in para.text) for para in q.paragraphs))
        # print(q.paragraphs[0].text)
        # break
        # text = flatten_iterable(x.text for x in q.paragraphs)
        # print(len(text))
        # print(text[0])
        # print(text[0][0])
        # print([" ".join([" ".join(s) for s in x.text]) for x in q.paragraphs])
    # for q in x:
    #     print(q.raw_question)
    #     response = requests.get("http://aristo-docker.dev.ai2:8087/decompose", {"text": q.raw_question})
    #     if response.status_code != 200:
    #         raise ValueError()
    #     decomposed = response.json()["question"]
    #     response.close()
    #
    #     selections = decomposed["selections"]
    #     if len(selections) != 4:
    #         raise ValueError()
    #     if [x["key"] for x in selections] != ["A", "B", "C", "D"]:
    #         raise ValueError()
    #
    #     query = {"searchQuery": decomposed, "maxResults": 5, "sourceTypes": ["elasticsearch"]}
    #     response = requests.post("http://aristo-background-knowledge.dev.ai2:8091/search", json=query)
    #     if response.status_code != 200:
    #         raise ValueError()
    #
    #     hits = response.json()
    #     response.close()
    #     print(hits["resultsPerSourceType"])
    #     hits = hits["resultsPerSourceType"]["elasticsearch"]
    #     print(len(hits))
    #     break


if __name__ == "__main__":
    main()
