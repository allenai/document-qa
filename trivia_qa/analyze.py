import numpy as np

from trivia_qa.build_span_corpus import TriviaQaWebDataset, TriviaQaSampleWebDataset


def build_doc_csv(data, output_file, sample=None):
    train = data.get_train()
    if sample is not None:
        np.random.shuffle(train)
        train = train[:sample]

    corpus = data.corpus
    feature_names = ["quid", "doc_id", "first_answer_token", "last_answer_token",
                     "first_answer_paragraph", "last_answer_paragraph", "n_answers", "doc_len"]
    stats = []

    n_no_answers = 0
    for question in train:
        for doc in question.web_docs:
            if doc.answer_spans is None:
                continue
            if len(doc.answer_spans) == 0:
                n_no_answers += 1
            else:
                text = corpus.get_document(doc.doc_id)
                para_lens = [sum(len(s) for s in para) for para in text]
                first_answer_token = sum(para_lens)
                last_answer_token = -1
                for p,s,e in doc.answer_spans:
                    token_start = sum(para_lens[:p]) + s
                    first_answer_token = min(first_answer_token, token_start)
                    last_answer_token = max(last_answer_token, token_start)

                stats.append([question.question, doc.doc_id, first_answer_token, last_answer_token,
                             np.min(doc.answer_spans[:, 0]), np.max(doc.answer_spans[:, 0]),
                              len(doc.answer_spans), sum(para_lens)])

    with open(output_file, "w") as f:
        f.write("\t".join(feature_names))
        f.write("\n")
        for line in stats:
            f.write("\t".join(str(x) for x in line))
            f.write("\n")


def show_first_match_doc(data):
    print("Loading train...")
    train = data.get_train()
    print("Checking...")
    corpus = data.corpus
    count = 0
    total = 0
    any_answers = 0
    np.random.shuffle(train)
    for question in train:
        # if any(np.any(doc.answer_spans[:, ]) > 0 for doc in question.web_docs):
        #     any_answers += 1
        cur_count = count
        for doc in question.web_docs:
            if len(doc.answer_spans) == 0:
                continue
            total += 1
            # near_answers = doc.answer_spans[doc.answer_spans[:, 4] < 1]
            if np.any(doc.answer_spans[:, 4] < 15 ):
                count += 1
                # text = flatten_iterable(flatten_iterable(corpus.get_document(doc.doc_id)))
                # print(" ".join(question.question))
                # print(" ".join(text[:15]))
                # input()
        if count > cur_count:
            any_answers += 1

    print("%d / %d (%.3f)" % (count, total, count/total))
    print("%d / %d (%.3f)" % (any_answers, len(train), any_answers / len(train)))

if __name__ == "__main__":
    show_first_match_doc(TriviaQaSampleWebDataset())
    # build_doc_csv(TriviaQaSampleWebDataset(), "/tmp/stats.tsv", sample=5000)


