import numpy as np

from data_processing.paragraph_qa import split_docs
from squad.squad import SquadCorpus


def main():
    docs = SquadCorpus().get_train_docs()
    titles = {x.doc_id:x.wiki_title for x in docs}

    all_questions = split_docs(docs)

    np.random.shuffle(all_questions)

    coref = {"it", "he", "she", "his", "her", "they", "this"}
    n_coref = 0
    for question in all_questions:
        if any(x.lower() in coref for x in question.question):
            print(titles[question.article_id])
            print(" ".join(question.question))
            print(str(set(x.text for x in question.answer)))
            input()
            n_coref += 1

    print("%d/%d (%.4f)" % (n_coref, len(all_questions), n_coref/len(all_questions)))


if __name__ == "__main__":
    main()
