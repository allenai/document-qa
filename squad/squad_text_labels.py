from typing import List
from collections import Counter
import numpy as np
from tqdm import tqdm

from data_processing.paragraph_qa import DocParagraphAndQuestion, Document, Paragraph, ContextLenBucketedKey, \
    ContextLenKey
from data_processing.preprocessed_corpus import Preprocessor, PreprocessedData
from data_processing.qa_data import ParagraphAndQuestionDatasetBuilder
from data_processing.span_data import ParagraphSpans, TokenSpans
from dataset import ClusteredBatcher
from squad.build_dataset import SquadCorpus
from squad.squad_official_evaluation import normalize_answer, exact_match_score
from trivia_qa.build_span_corpus import TriviaQaWebDataset
from trivia_qa.triviaqa_training_data import ExtractSingleParagraph, InMemoryWebQuestionBuilder
from utils import flatten_iterable


"""
squad where the labels are derived from the answer text, ignoring the "correct span" marked by users.
In other words all spans with the correct text, not just the user's spans, are marked as correct. 
On one hand this is throwing out the information about which span might be the most informative place to focus on, 
on the other hand this aligns with the test-time metrics which are text based and m
"""

def _squad_answer_detector(paragraph: Paragraph,
                          normalized_text: List[str],
                          tagged_spans: ParagraphSpans):
    correct_answer_text = [normalize_answer(x.text) for x in tagged_spans]

    answer_spans = []
    n_words = paragraph.n_context_words
    for ix in range(n_words):
        word = normalized_text[ix]
        for answer in correct_answer_text:
            span_text = word
            end_ix = ix
            any_found = False
            while True:
                if span_text == answer:
                    answer_spans.append((ix, end_ix))
                    # continue in case the span including the next token also matches when normalized
                elif not answer.startswith(span_text):
                    break
                end_ix += 1
                if end_ix == n_words:
                    break
                next_token = normalized_text[end_ix]
                if next_token not in answer[len(span_text):]:
                    break
                span_text = normalize_answer(paragraph.get_original_text(ix, end_ix))
            if any_found:
                answer_spans.append((ix, end_ix))
                break

    for x in tagged_spans:
        start, end = x.para_word_start, x.para_word_end
        if (start, end) not in answer_spans:
            extracted = normalize_answer(paragraph.get_original_text(start, end))
            if any(extracted == x for x in correct_answer_text):
                raise RuntimeError("Missed an answer span!")  # Sanity check, we should have extracted this
            else:
                # normally due to the correct text being cut off mid word, or otherwise text that does not
                # land between our tokens
                # in this case we will just include the tagged span as training data anyway
                answer_spans.append((start, end))

    if len(answer_spans) == 0:
        raise ValueError()
    else:
        return np.array(answer_spans, dtype=np.int32)


def preprocess(data: List[Document]) -> List[DocParagraphAndQuestion]:
    out = []
    for doc in data:
        for paragraph in doc.paragraphs:
            # Precomputed once per each paragraph to speed up our answer detection algorithm
            normalized_text = [normalize_answer(paragraph.get_original_text(i, i))
                               for i in range(paragraph.n_context_words)]
            for q in paragraph.questions:
                spans = _squad_answer_detector(paragraph, normalized_text, q.answer)
                out.append(DocParagraphAndQuestion(q.words, TokenSpans([x.text for x in q.answer], spans), q.question_id, paragraph))
    return out


class TagTextAnswers(Preprocessor):
    def preprocess(self, data: List[Document], evidence):
        return preprocess(data)


def check_answers():
    data = SquadCorpus()
    computed = preprocess(tqdm(data.get_train(), desc="tagging"))
    for para in tqdm(computed, desc="checking"):
        for (start, end) in para.answer.answer_spans:
            text = para.paragraph.get_original_text(start, end)
            if not any(exact_match_score(x, text) for x in para.answer.answer_text):
                raise ValueError()


def count_answers():
    data = SquadCorpus()
    computed = preprocess(tqdm(data.get_dev(), desc="tagging"))
    counts = Counter([len(x.answer.answer_spans) for x in computed])

    for i in range(0, 11):
        print("%d: %d (%.4f)" % (i, counts[i], counts[i]/len(computed)))


def test_build_training_data():
    train_batching = ClusteredBatcher(60, ContextLenBucketedKey(3), True, False)
    eval_batching = ClusteredBatcher(60, ContextLenKey(), False, False)
    data = PreprocessedData(SquadCorpus(),
                            TagTextAnswers(),
                            ParagraphAndQuestionDatasetBuilder(train_batching, eval_batching),
                            eval_on_verified=False,
                            sample=20, sample_dev=20
                            # sample_dev=100, sample=100, eval_on_verified=False
                            )
    data.preprocess()
    data = data.get_train()
    for batch in data.get_epoch():
        for x in batch:
            print(x.answer.answer_spans.shape)


if __name__ == "__main__":
    count_answers()
    # test_build_training_data()