import numpy as np

from data_processing.span_data import SpanCorpus


def main():
    data = SpanCorpus("squad")
    train = data.get_train_docs()

    context_features = []
    answers = []

    for doc in train:
        for para in doc.paragraphs:
            context = para.context
            n_context_wprds = sum(len(x) for x in context)
            for question in para.questions:
                question_ans = []
                for ans in question.answer:
                    question_ans.append((ans.sent_start, ans.para_word_start,
                                         ans.word_start,
                                         ans.para_word_end-ans.para_word_start+1,
                                         ans.sent_end - ans.sent_start + 1,
                                         len(ans.text)
                                         ))
                answers.append(np.array(question_ans).min(axis=0))
                context_features.append(np.array((len(context), n_context_wprds, len(question.answer),
                                         len(context[np.array(question_ans).min(axis=0)[0]]))))

    with open("/tmp/answer_stats.csv", "w") as f:
        f.write(",".join(["n_context_sentences", "n_context_words", "n_answers", "n_answer_sentence_words",
                          "answer_sentence_num", "answer_word_num", "answer_sentence_word_num",
                          "n_answer_words", "n_answer_sentences", "n_answer_char"]))
        f.write("\n")
        for i in range(len(answers)):
            f.write(",".join(str(x) for x in np.concatenate([context_features[i], answers[i]])))
            f.write("\n")

if __name__ == "__main__":
    main()