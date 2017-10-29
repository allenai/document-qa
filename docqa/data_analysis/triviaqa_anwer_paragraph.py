import numpy as np

from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf
from docqa.data_processing.preprocessed_corpus import preprocess_par
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion


def main():
    data = TriviaQaOpenDataset()
    # data = TriviaQaWebDataset()
    print("Loading...")
    all_questions = data.get_dev()

    questions = [q for q in all_questions if any(len(x.answer_spans) > 0 for x in q.all_docs)]
    print("%d/%d (%.4f) have an answer" % (len(questions), len(all_questions), len(questions)/len(all_questions)))

    np.random.shuffle(questions)

    pre = ExtractMultiParagraphsPerQuestion(MergeParagraphs(400),
                                            TopTfIdf(NltkPlusStopWords(), 20),
                                            require_an_answer=False)
    print("Done")

    out = preprocess_par(questions[:2000], data.evidence, pre, 2, 1000)

    n_counts = np.zeros(20)
    n_any = np.zeros(20)
    n_any_all = np.zeros(20)

    for q in out.data:
        for i, p in enumerate(q.paragraphs):
            n_counts[i] += 1
            n_any[i] += len(p.answer_spans) > 0

        for i, p in enumerate(q.paragraphs):
            if len(p.answer_spans) > 0:
                n_any_all[i:] += 1
                break

    print(n_any_all / out.true_len)
    print(n_any/n_counts)
    print(n_counts)



if __name__ == "__main__":
    main()