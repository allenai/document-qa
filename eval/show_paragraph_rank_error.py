import argparse
import json
import numpy as np

from data_processing.text_utils import NltkPlusStopWords
from trivia_qa.build_span_corpus import TriviaQaWebDataset


class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('answer_file', help='name of output to exmaine')
    args = parser.parse_args()

    print("Loading answers...")
    with open(args.answer_file, "r") as f:
        data = json.load(f)

    print("Loading questions...")
    dataset = TriviaQaWebDataset()
    trivia_questions = dataset.get_train()
    trivia_questions = {q.question_id:q for q in trivia_questions}

    stop = NltkPlusStopWords(punctuation=True).words
    np.random.shuffle(data)

    for point in data:
        q = trivia_questions[point["quid"]]
        doc = [d for d in q.all_docs if d.doc_id == point["doc_id"]][0]
        s, e = point["spans"][0]
        if not np.any(np.logical_and(doc.answer_spans[:, 0] >= s, doc.answer_spans[:, 1] < e)):
            q_words = set(x.lower() for x in q.question if x.lower() not in stop)
            doc_text = dataset.evidence.get_document(doc.doc_id, None, flat=True)
            text = doc_text[s:e]
            print(" ".join(q.question))
            for ix, word in enumerate(text):
                if word.lower() in q_words:
                    text[ix] = bcolors.CORRECT + word + bcolors.ENDC
            print(" ".join(text))

            print()
            s,e = doc.answer_spans[0]
            doc_text[s] = bcolors.ERROR + doc_text[s]
            doc_text[e] = doc_text[e] +  bcolors.ENDC
            text = doc_text[max(s-100, 0):min(e+100, len(doc_text))]
            for ix, word in enumerate(text):
                if word.lower() in q_words:
                    text[ix] = bcolors.CORRECT + word + bcolors.ENDC
            print(" ".join(text))
            input()


if __name__ == "__main__":
    main()