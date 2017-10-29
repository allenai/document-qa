import argparse

from os.path import exists

from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset
from docqa.triviaqa.evidence_corpus import get_evidence_voc

"""
Build vocab of all words in the triviaqa dataset, including
all documents and all train questions.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("-m", "--min_count", type=int, default=1)
    parser.add_argument("-n", "--n_processes", type=int, default=1)
    args = parser.parse_args()

    if exists(args.output):
        raise ValueError()

    data = TriviaQaOpenDataset()
    corpus_voc = get_evidence_voc(data.evidence, args.n_processes)

    print("Adding question voc...")
    train = data.get_train()
    for q in train:
        corpus_voc.update(q.question)

    print("Saving...")
    with open(args.output, "w") as f:
        for word, c in corpus_voc.items():
            if c >= args.min_count:
                f.write(word)
                f.write("\n")


if __name__ == "__main__":
    main()