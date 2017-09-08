from collections import Counter

import numpy as np
from tqdm import tqdm

from data_processing.document_splitter import MergeParagraphs, ContainsQuestionWord, DocumentSplitter, \
    ExtractedParagraphWithAnswers
from data_processing.preprocessed_corpus import PreprocessedData
from data_processing.qa_data import Batcher
from data_processing.text_utils import NltkPlusStopWords, WordNormalizer
from paragraph_selection.paragraph_selection_featurizer import NGramMatchingFeaturizer, ParagraphOrderFeatures, \
    ParagraphFeatures
from experimental.paragraph_selection.paragraph_selection_model import SelectionDatasetBuilder, ParagraphSelectionFeaturizer
from trivia_qa.build_span_corpus import TriviaQaOpenDataset, TriviaQaWebDataset, TriviaQaSampleWebDataset
from trivia_qa.read_data import SearchDoc, TriviaQaQuestion
from utils import flatten_iterable


def basic_stats(corpus):
    train = corpus.get_train()
    n_docs = sum(len(q.all_docs) for q in train)
    n_with_answer = sum(sum(len(doc.answer_spans) > 0 for doc in q.all_docs) for q in train)
    print(n_docs)
    print(n_with_answer)


def paragraph_stats(corpus, splitter: DocumentSplitter, sample):
    stop = NltkPlusStopWords(punctuation=True).words

    if "," not in stop:
        raise ValueError()

    norm = WordNormalizer()
    data = corpus.get_dev()
    pairs = flatten_iterable([(q, doc) for doc in q.all_docs] for q in data)
    data = [pairs[i] for i in np.random.choice(np.arange(0, len(pairs)), sample, replace=False)]

    word_matches = Counter()
    n_para = []
    n_answers = []
    n_question_words = []

    for q,doc in data:
        if len(doc.answer_spans) == 0:
            continue
        q_words = set(x.lower() for x in q.question)
        q_words -= stop
        # q_words = set(norm.normalize(w) for w in q_words)

        text = corpus.evidence.get_document(doc.doc_id)
        para = splitter.split_annotated(text, doc.answer_spans)
        n_para.append(len(para))
        n_answers += [len(x.answer_spans) for x in para]

        for x in para:
            match_set = set()
            n_matches = 0
            text = flatten_iterable(x.text)
            for word in text:
                word = word.lower()
                if word in q_words:
                    n_matches += 1
                    match_set.add(word)
            if len(match_set) == 0 and len(x.answer_spans) > 0:
                print_paragraph(q, x)
                input()
            word_matches.update(match_set)
            n_question_words.append(n_matches)
        # n_question_words += [sum(norm.normalize(w) in q_words for w in flatten_iterable(x.text)) for x in para]
        # n_question_words += [sum(w.lower() in q_words for w in flatten_iterable(x.text)) for x in para]

    n_answers = np.array(n_answers)
    n_question_words = np.array(n_question_words)
    any_answers = n_answers > 0
    any_question_word = n_question_words > 0

    total_para = len(any_answers)
    total_q = len(n_para)

    no_quesiton_and_answer = any_answers[np.logical_not(any_question_word)]

    print("%d/%d (%.4f) pairs have an answer" % (total_q, len(data), total_q/len(data)))
    print("%d para in %d questions (av %.4f)" % (sum(n_para), total_q, sum(n_para)/total_q))
    print("%d/%d (%.4f) paragraphs have answers" % (any_answers.sum(), total_para, any_answers.mean()))
    print("%d/%d (%.4f) paragraphs have question word" % (any_question_word.sum(), total_para, any_question_word.mean()))
    print("%d/%d (%.4f) no question words have answers" % (no_quesiton_and_answer.sum(),
                                                           len(no_quesiton_and_answer),
                                                           no_quesiton_and_answer.mean()))
    # for k,v in word_matches.most_common(100):
    #     print("%s: %d" % (k, v))


def print_paragraph(question: TriviaQaQuestion, para: ExtractedParagraphWithAnswers):
    print(" ".join(question.question))
    print(question.answer.all_answers)
    context = flatten_iterable(para.text)
    for s,e in para.answer_spans:
        context[s] = "{{{" + context[s]
        context[e] = context[e] + "}}}"
    print(" ".join(context))


def print_questions(question, answers, context, answer_span):
    print(" ".join(question))
    print(answers)
    context = flatten_iterable(context)
    for s,e in answer_span:
        context[s] = "{{{" + context[s]
        context[e] = context[e] + "}}}"
    print(" ".join(context))


def contains_question_word():
    data = TriviaQaWebDataset()
    stop = NltkPlusStopWords(punctuation=True).words
    doc_filter = ContainsQuestionWord(NltkPlusStopWords(punctuation=True))
    splits = MergeParagraphs(400)
    # splits = Truncate(400)
    questions = data.get_dev()
    pairs = flatten_iterable([(q, doc) for doc in q.all_docs] for q in questions)
    pairs.sort(key=lambda x: (x[0].question_id, x[1].doc_id))
    np.random.RandomState(0).shuffle(questions)
    has_token = 0
    total = 0
    used = Counter()

    for q, doc in tqdm(pairs[:1000]):
        text = data.evidence.get_document(doc.doc_id, splits.reads_first_n)
        q_tokens = set(x.lower() for x in q.question)
        q_tokens -= stop
        for para in splits.split_annotated(text, doc.answer_spans):
            # if para.start == 0:
            #     continue
            if len(para.answer_spans) == 0:
                continue
            if any(x.lower() in q_tokens for x in flatten_iterable(para.text)):
                has_token += 1
                for x in flatten_iterable(para.text):
                    if x in q_tokens:
                        used[x] += 1
            # else:
            #     print_questions(q.question, q.answer.all_answers, para.text, para.answer_spans)
            #     input()
            total += 1
    for k,v in used.most_common(200):
        print("%s: %d" % (k, v))
    print(has_token/total)


def show_questions():
    corpus = TriviaQaWebDataset()
    questions = corpus.get_train()
    np.random.shuffle(questions)
    for q in questions[:500]:
        print(" ".join(q.question))


def prep():
    corpus = TriviaQaWebDataset()
    dev = corpus.get_dev()
    norm = WordNormalizer(stemmer="wordnet")
    fe = ParagraphSelectionFeaturizer(MergeParagraphs(400), None,
                                                [
                                                    NGramMatchingFeaturizer(None, norm, (1, 1)),
                                                    # NGramFineGrained(stop, norm, [(1, (True, True, True)),
                                                    #                               (2, (True, True, True))]),
                                                 ],
                                                [ParagraphOrderFeatures(), ParagraphFeatures()],
                                                filter_initial_zeros=False,
                                                prune_no_answer=True)
    fe.preprocess(dev, corpus.evidence)

if __name__ == "__main__":
    show_questions()
    # contains_question_word()
    # paragraph_stats(TriviaQaWebDataset(), MergeParagraphs(400), 500)
