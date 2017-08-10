from typing import List

import numpy as np

from data_processing.qa_data import Answer

"""
Utility methods for dealing with span data, and span question answers
"""


def span_len(span):
    return span[1] - span[0] + 1


def span_f1(true_span, pred_span):
    start = max(true_span[0], pred_span[0])
    stop = min(true_span[1], pred_span[1])
    if start > stop:
        return 0
    overlap_len = span_len((start, stop))
    p = overlap_len / span_len(pred_span)
    r = overlap_len / span_len(true_span)
    return 2 * p * r / (p + r)


def get_best_span(word_start_probs, word_end_probs):
    max_val = -1
    best_word_span = None

    span_start = -1
    span_start_val = -1

    for word_ix in range(0, len(word_start_probs)):

        # Move `span_start` forward iff that would improve our score
        # Thus span_start will always be the largest valued start between
        # [0, `word_ix`]
        if span_start_val < word_start_probs[word_ix]:
            span_start_val = word_start_probs[word_ix]
            span_start = word_ix

        # Check if the new span is the best one yet
        if span_start_val * word_end_probs[word_ix] > max_val:
            best_word_span = (span_start, word_ix)
            max_val = span_start_val * word_end_probs[word_ix]

    return best_word_span, max_val


def get_best_span_bounded(word_start_probs, word_end_probs, bound):
    max_val = -1
    best_word_span = None

    span_start = -1
    span_start_val = -1

    for word_ix in range(0, len(word_start_probs)):

        # Move `span_start` forward iff that would improve our score
        if span_start_val < word_start_probs[word_ix]:
            span_start_val = word_start_probs[word_ix]
            span_start = word_ix

        # Jump to the next largest span start iff we reach the boundary limit
        if (word_ix - span_start + 1) > bound:
            span_start += 1 + np.argmax(word_start_probs[span_start+1:word_ix+1])
            # span_start = np.argmax(word_start_probs[span_start + 1:word_ix + 1])
            span_start_val = word_start_probs[span_start]

        # Check if the new span is the best one yet
        if span_start_val * word_end_probs[word_ix] > max_val:
            best_word_span = (span_start, word_ix)
            max_val = span_start_val * word_end_probs[word_ix]

    return best_word_span, max_val


def get_best_in_sentence_span(start_probs, end_probs, sent_lens):
    max_val = -1
    best_word_span = None
    span_start = 0
    span_start_val = start_probs[0]
    on_sent = 0
    sent_end = sent_lens[0]
    for word_ix in range(0, len(start_probs)):
        if word_ix == sent_end:
            # reached the start of the next sentence, reset the span_start pointer
            on_sent += 1
            if on_sent >= len(sent_lens):
                break
            sent_end += sent_lens[on_sent]
            span_start = word_ix
            span_start_val = start_probs[word_ix]
        else:
            if span_start_val < start_probs[word_ix]:
                span_start_val = start_probs[word_ix]
                span_start = word_ix

        # Check if the new span is the best one yet
        if span_start_val * end_probs[word_ix] > max_val:
            best_word_span = (span_start, word_ix)
            max_val = span_start_val * end_probs[word_ix]

    return best_word_span, max_val


def get_best_span_from_sent_predictions(per_sent_start_pred, per_sent_end_pred, sent_lens):
    max_val = -1
    best_word_span = None
    word_offset = 0

    # Min in case the the # of sentences was truncated
    for sent_ix in range(min(len(sent_lens), len(per_sent_start_pred))):
        sent_len = sent_lens[sent_ix]
        start_pred = per_sent_start_pred[sent_ix]
        end_pred = per_sent_end_pred[sent_ix]

        span_start = 0
        span_start_val = start_pred[0]

        # Min in case the # of sentence predictions was truncated
        for word_ix in range(0, min(sent_len, len(start_pred))):
            if span_start_val < start_pred[word_ix]:
                span_start_val = start_pred[word_ix]
                span_start = word_ix

            if span_start_val * end_pred[word_ix] > max_val:
                best_word_span = (word_offset + span_start, word_offset + word_ix)
                max_val = span_start_val * end_pred[word_ix]
        word_offset += sent_len

    return best_word_span, max_val


def compute_span_f1(true_span, pred_span):
    start = max(true_span[0], pred_span[0])
    stop = min(true_span[1], pred_span[1])
    if start > stop:
        return 0
    overlap_len = span_len((start, stop))
    p = overlap_len / span_len(pred_span)
    r = overlap_len / span_len(true_span)
    return 2 * p * r / (p + r)



class ParagraphSpan(object):
    def __init__(self,
                 sent_start: int, word_start: int, char_start: int,
                 sent_end: int, word_end: int, char_end: int,
                 para_word_start: int, para_word_end: int,
                 text: str):
        self.sent_start = sent_start
        self.word_start = word_start
        self.char_start = char_start
        self.sent_end = sent_end
        self.word_end = word_end
        self.char_end = char_end
        self.para_word_start = para_word_start
        self.para_word_end = para_word_end
        self.text = text

    def as_tuple(self):
        return self.sent_start, self.word_start, self.char_start, self.sent_end, \
               self.word_end, self.char_end, self.para_word_start, self.para_word_end, self.text


class ParagraphSpans(Answer):
    def __init__(self, spans: List[ParagraphSpan]):
        self.spans = spans

    def to_json_object(self):
        return [x.as_tuple() for x in self.spans]

    def get_vocab(self):
        return []

    def __getitem__(self, item):
        return self.spans[item]

    def __iter__(self):
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)
