import argparse
import pickle

from docqa.data_processing.word_vectors import load_word_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vecs")
    parser.add_argument("vocab")
    parser.add_argument("output")
    args = parser.parse_args()

    voc = set()
    with open(args.vocab) as f:
        for line in f:
            voc.add(line.strip())

    voc = load_word_vectors(args.vecs, voc)
    with open(args.output, "wb") as f:
        pickle.dump(voc, f)


if __name__ == "__main__":
    main()