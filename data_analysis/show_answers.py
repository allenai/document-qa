import json
import numpy as np
from os.path import join

from config import TRIVIA_QA
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.read_data import normalize_wiki_filename


def main():
    dataset = TriviaQaWebDataset()
    with open("/Users/chrisc/predictions.json") as f:
        pred = json.load(f)

    fns = {}
    print("Loading proper filenames")
    source = join(TRIVIA_QA, "qa", "web-test-without-answers.json")

    with open(join(source)) as f:
        data = json.load(f)["Data"]

    for point in data:
        for doc in point["EntityPages"]:
            filename = doc["Filename"]
            fn = join("wikipedia", filename[:filename.rfind(".")])
            fn = normalize_wiki_filename(fn)
            fns[(point["QuestionId"], fn)] = filename

    questions = dataset.get_test()
    np.random.shuffle(questions)
    for point in dataset.get_test():
        print(point.question)
        print(" ".join(point.question))
        for doc in point.all_docs:
            filename = dataset.evidence.file_id_map[doc.doc_id]
            if filename.startswith("web"):
                true_name = filename[4:] + ".txt"
            else:
                true_name = fns[(point.question_id, filename)]
            key = point.question_id + "--" + true_name
            ans = pred[key]
            print(ans)
        input()


if __name__ == "__main__":
    main()