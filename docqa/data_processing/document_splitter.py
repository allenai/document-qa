from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from docqa.utils import flatten_iterable

from docqa.data_processing.text_utils import NltkPlusStopWords, ParagraphWithInverse
from docqa.configurable import Configurable
from docqa.triviaqa.evidence_corpus import TriviaQaEvidenceCorpusTxt

"""
Splits a document into paragraphs
"""


class ExtractedParagraph(object):
    __slots__ = ["text", "start", "end"]

    def __init__(self, text: List[List[str]], start: int, end: int):
        """
        :param text: List of source paragraphs that have been merged to form `self`
        :param start: start token of this text in the source document
        :param end: end token of this text in the source document
        """
        self.text = text
        self.start = start
        self.end = end

    @property
    def n_context_words(self):
        return sum(len(s) for s in self.text)


class ExtractedParagraphWithAnswers(ExtractedParagraph):
    __slots__ = ["answer_spans"]

    def __init__(self, text: List[List[str]], start: int, end: int, answer_spans: np.ndarray):
        super().__init__(text, start, end)
        self.answer_spans = answer_spans


class DocParagraphWithAnswers(ExtractedParagraphWithAnswers):
    __slots__ = ["doc_id"]

    def __init__(self, text: List[List[str]], start: int, end: int, answer_spans: np.ndarray,
                 doc_id):
        super().__init__(text, start, end, answer_spans)
        self.doc_id = doc_id


class ParagraphFilter(Configurable):
    """ Selects and ranks paragraphs """

    def prune(self, question, paragraphs: List[ExtractedParagraph]) -> List[ExtractedParagraph]:
        raise NotImplementedError()


class FirstN(ParagraphFilter):
    def __init__(self, n):
        self.n = n

    def prune(self, question, paragraphs: List[ExtractedParagraphWithAnswers]):
        return sorted(paragraphs, key=lambda x: x.start)[:self.n]


class ContainsQuestionWord(ParagraphFilter):
    def __init__(self, stop, allow_first=True, n_paragraphs: int=None):
        self.stop = stop
        self.allow_first = allow_first
        self.n_paragraphs = n_paragraphs

    def prune(self, question, paragraphs: List[ExtractedParagraphWithAnswers]):
        q_words = {x.lower() for x in question}
        q_words -= self.stop.words
        output = []

        for para in paragraphs:
            if self.allow_first and para.start == 0:
                output.append(para)
                continue
            keep = False
            for sent in para.text:
                if any(x.lower() in q_words for x in sent):
                    keep = True
                    break
            if keep:
                output.append(para)
        if self.n_paragraphs is not None:
            output = output[:self.n_paragraphs]
        return output


class TopTfIdf(ParagraphFilter):
    def __init__(self, stop, n_to_select: int, filter_dist_one: bool=False, rank=True):
        self.stop = stop
        self.rank = rank
        self.n_to_select = n_to_select
        self.filter_dist_one = filter_dist_one

    def prune(self, question, paragraphs: List[ExtractedParagraph]):
        if not self.filter_dist_one and len(paragraphs) == 1:
            return paragraphs

        tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self.stop.words)
        text = []
        for para in paragraphs:
            text.append(" ".join(" ".join(s) for s in para.text))
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError:
            return []

        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        sorted_ix = np.lexsort(([x.start for x in paragraphs], dists))  # in case of ties, use the earlier paragraph

        if self.filter_dist_one:
            return [paragraphs[i] for i in sorted_ix[:self.n_to_select] if dists[i] < 1.0]
        else:
            return [paragraphs[i] for i in sorted_ix[:self.n_to_select]]

    def dists(self, question, paragraphs: List[ExtractedParagraph]):
        tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self.stop.words)
        text = []
        for para in paragraphs:
            text.append(" ".join(" ".join(s) for s in para.text))
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError:
            return []

        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        sorted_ix = np.lexsort(([x.start for x in paragraphs], dists))  # in case of ties, use the earlier paragraph

        if self.filter_dist_one:
            return [(paragraphs[i], dists[i]) for i in sorted_ix[:self.n_to_select] if dists[i] < 1.0]
        else:
            return [(paragraphs[i], dists[i]) for i in sorted_ix[:self.n_to_select]]


class ShallowOpenWebRanker(ParagraphFilter):
    # Hard coded weight learned from a logistic regression classifier
    TFIDF_W = 5.13365065
    LOG_WORD_START_W = 0.46022765
    FIRST_W = -0.08611607
    LOWER_WORD_W = 0.0499123
    WORD_W = -0.15537181

    def __init__(self, n_to_select):
        self.n_to_select = n_to_select
        self._stop = NltkPlusStopWords(True).words
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self._stop)

    def get_features(self, question: List[str], paragraphs: List[List[ExtractedParagraphWithAnswers]]):
        scores = self.score_paragraphs(question, flatten_iterable(paragraphs))
        # return scores
        return np.expand_dims(scores, 1)

    def get_feature_names(self):
        return ["Score"]

    def score_paragraphs(self, question, paragraphs: List[ExtractedParagraphWithAnswers]):
        tfidf = self._tfidf
        text = []
        for para in paragraphs:
            text.append(" ".join(" ".join(s) for s in para.text))
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError:
            return []

        q_words = {x for x in question if x.lower() not in self._stop}
        q_words_lower = {x.lower() for x in q_words}
        word_matches_features = np.zeros((len(paragraphs), 2))
        for para_ix, para in enumerate(paragraphs):
            found = set()
            found_lower = set()
            for sent in para.text:
                for word in sent:
                    if word in q_words:
                        found.add(word)
                    elif word.lower() in q_words_lower:
                        found_lower.add(word.lower())
            word_matches_features[para_ix, 0] = len(found)
            word_matches_features[para_ix, 1] = len(found_lower)

        tfidf = pairwise_distances(q_features, para_features, "cosine").ravel()
        starts = np.array([p.start for p in paragraphs])
        log_word_start = np.log(starts/400.0 + 1)
        first = starts == 0
        scores = tfidf * self.TFIDF_W + self.LOG_WORD_START_W * log_word_start + self.FIRST_W * first +\
                 self.LOWER_WORD_W * word_matches_features[:, 1] + self.WORD_W * word_matches_features[:, 0]
        return scores

    def prune(self, question, paragraphs: List[ExtractedParagraphWithAnswers]):
        scores = self.score_paragraphs(question, paragraphs)
        sorted_ix = np.argsort(scores)

        return [paragraphs[i] for i in sorted_ix[:self.n_to_select]]

    def __getstate__(self):
        return dict(n_to_select=self.n_to_select)

    def __setstate__(self, state):
        return self.__init__(state['n_to_select'])


class DocumentSplitter(Configurable):
    """ Re-organize a collection of tokenized paragraphs into `ExtractedParagraph`s """

    @property
    def max_tokens(self):
        """ max number of tokens a paragraph from this splitter can have, or None """
        return None

    @property
    def reads_first_n(self):
        """ only requires the first `n` tokens of the documents, or None """
        return None

    def split(self, doc: List[List[List[str]]]) -> List[ExtractedParagraph]:
        """
        Splits a list paragraphs->sentences->words to a list of `ExtractedParagraph`
        """
        raise NotImplementedError()

    def split_annotated(self, doc: List[List[List[str]]], spans: np.ndarray) -> List[ExtractedParagraphWithAnswers]:
        """
        Split a document and additionally splits answer_span of each paragraph
        """
        out = []
        for para in self.split(doc):
            para_spans = spans[np.logical_and(spans[:, 0] >= para.start, spans[:, 1] < para.end)] - para.start
            out.append(ExtractedParagraphWithAnswers(para.text, para.start, para.end, para_spans))
        return out

    def split_inverse(self, paras: List[ParagraphWithInverse], delim="\n") -> List[ParagraphWithInverse]:
        """
        Split a document consisting of `ParagraphWithInverse` objects
        `delim` will be added to the original_txt of between each paragraph
        """
        full_para = ParagraphWithInverse.concat(paras, delim)

        split_docs = self.split([x.text for x in paras])

        out = []
        for para in split_docs:
            # Grad the correct inverses and convert back to the paragraph level
            inv = full_para.spans[para.start:para.end]
            text = full_para.get_original_text(para.start, para.end-1)
            inv -= inv[0][0]
            out.append(ParagraphWithInverse(para.text, text, inv))
        return out


class Truncate(DocumentSplitter):
    """ map a document to a single paragraph of the first `max_tokens` tokens """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def max_tokens(self):
        return self.max_tokens

    @property
    def reads_first_n(self):
        return self.max_tokens

    def split(self, doc: List[List[List[str]]]):
        output = []
        cur_tokens = 0
        for para in doc:
            for sent in para:
                if cur_tokens + len(sent) > self.max_tokens:
                    output.append(sent[:self.max_tokens - cur_tokens])
                    return [ExtractedParagraph(output, 0, self.max_tokens)]
                else:
                    cur_tokens += len(sent)
                    output.append(sent)
        return [ExtractedParagraph(output, 0, cur_tokens)]


class MergeParagraphs(DocumentSplitter):
    """
    Merge paragraphs up to a maximum size. Paragraphs of a larger size will be truncated.
    """

    def __init__(self, max_tokens: int, top_n: int=None):
        self.max_tokens = max_tokens
        self.top_n = top_n

    @property
    def reads_first_n(self):
        return self.top_n

    def max_tokens(self):
        return self.max_tokens

    def split(self, doc: List[List[List[str]]]):
        all_paragraphs = []

        on_doc_token = 0  # the word in the document the current paragraph starts at
        on_paragraph = []  # text we have collect for the current paragraph
        cur_tokens = 0   # number of tokens in the current paragraph

        word_ix = 0
        for para in doc:
            para = flatten_iterable(para)
            n_words = len(para)
            if self.top_n is not None and (word_ix+self.top_n)>self.top_n:
                if word_ix == self.top_n:
                    break
                para = para[:self.top_n - word_ix]
                n_words = self.top_n - word_ix

            start_token = word_ix
            end_token = start_token + n_words
            word_ix = end_token

            if cur_tokens + n_words > self.max_tokens:
                if cur_tokens != 0:  # end the current paragraph
                    all_paragraphs.append(ExtractedParagraph(on_paragraph, on_doc_token, start_token))
                    on_paragraph = []
                    cur_tokens = 0

                if n_words >= self.max_tokens:  # either truncate the given paragraph, or begin a new paragraph
                    text = para[:self.max_tokens]
                    all_paragraphs.append(ExtractedParagraph([text], start_token,
                                                             start_token + self.max_tokens))
                    on_doc_token = end_token
                else:
                    on_doc_token = start_token
                    on_paragraph.append(para)
                    cur_tokens = n_words
            else:
                on_paragraph.append(para)
                cur_tokens += n_words

        if len(on_paragraph) > 0:
            all_paragraphs.append(ExtractedParagraph(on_paragraph, on_doc_token, word_ix))

        return all_paragraphs


class PreserveParagraphs(DocumentSplitter):
    """
    Convience class that preserves the document's natural paragraph delimitation
    """
    def split(self, doc: List[List[List[str]]]):
        out = []
        on_token = 0
        for para in doc:
            flattened_para = flatten_iterable(para)
            end = on_token + len(flattened_para)
            out.append(ExtractedParagraph([flatten_iterable(para)], on_token, end))
            on_token = end
        return out


def extract_tokens(paragraph: List[List[str]], n_tokens) -> List[List[str]]:
    output = []
    cur_tokens = 0
    for sent in paragraph:
        if len(sent) + cur_tokens > n_tokens:
            if n_tokens != cur_tokens:
                output.append(sent[:n_tokens - cur_tokens])
            return output
        else:
            output.append(sent)
            cur_tokens += len(sent)
    return output


def test_splitter(splitter: DocumentSplitter, n_sample, n_answer_spans, seed=None):
    rng = np.random.RandomState(seed)
    corpus = TriviaQaEvidenceCorpusTxt()
    docs = sorted(corpus.list_documents())
    rng.shuffle(docs)
    max_tokens = splitter.max_tokens
    read_n = splitter.reads_first_n
    for doc in docs[:n_sample]:
        text = corpus.get_document(doc, read_n)
        fake_answers = []
        offset = 0
        for para in text:
            flattened = flatten_iterable(para)
            fake_answer_starts = np.random.choice(len(flattened), min(len(flattened)//2, np.random.randint(5)), replace=False)
            max_answer_lens = np.minimum(len(flattened) - fake_answer_starts, 30)
            fake_answer_ends = fake_answer_starts + np.floor(rng.uniform() * max_answer_lens).astype(np.int32)
            fake_answers.append(np.concatenate([np.expand_dims(fake_answer_starts, 1), np.expand_dims(fake_answer_ends, 1)], axis=1) + offset)
            offset += len(flattened)

        fake_answers = np.concatenate(fake_answers, axis=0)
        flattened = flatten_iterable(flatten_iterable(text))
        answer_strs = set(tuple(flattened[s:e+1]) for s,e in fake_answers)

        paragraphs = splitter.split_annotated(text, fake_answers)

        for para in paragraphs:
            text = flatten_iterable(para.text)
            if max_tokens is not None and len(text) > max_tokens:
                raise ValueError("Paragraph len len %d, but max tokens was %d" % (len(text), max_tokens))
            start, end = para.start, para.end
            if text != flattened[start:end]:
                raise ValueError("Paragraph is missing text, given bounds were %d-%d" % (start, end))
            for s, e in para.answer_spans:
                if tuple(text[s:e+1]) not in answer_strs:
                    print(s,e)
                    raise ValueError("Incorrect answer for paragraph %d-%d (%s)" % (start, end, " ".join(text[s:e+1])))


def show_paragraph_lengths():
    corpus = TriviaQaEvidenceCorpusTxt()
    docs = corpus.list_documents()
    np.random.shuffle(docs)
    para_lens = []
    for doc in docs[:5000]:
        text = corpus.get_document(doc)
        para_lens += [sum(len(s) for s in x) for x in text]
    para_lens = np.array(para_lens)
    for i in [400, 500, 600, 700, 800]:
        print("Over %s: %.4f" % (i, (para_lens > i).sum()/len(para_lens)))


if __name__ == "__main__":
    test_splitter(MergeParagraphs(200), 1000, 20, seed=0)
    # show_paragraph_lengths()




