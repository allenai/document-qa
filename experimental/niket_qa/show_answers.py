from experimental.niket_qa.build_niket_dataset import NiketCorpus


def main():
    data = NiketCorpus()
    test = data.get_test()
    n_empty = 0
    for point in test:
        print(" ".join(point.question))
        print(point.answer.answer_text)
        if len(point.answer.answer_text) == 0:
            n_empty += 1

    print(n_empty, len(test), n_empty/len(test))


if __name__ == "__main__":
    main()