"""
Script to split a document into paragraphs
"""
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from configurable import Configurable
from trivia_qa.evidence_corpus import TriviaQaEvidenceCorpusTxt
from utils import flatten_iterable


class ExtractedParagraph(object):
    __slots__ = ["text", "start", "end", "answer_spans"]

    def __init__(self, text: List[List[str]], start: int, end: int, answer_spans: np.ndarray):
        self.text = text
        self.start = start
        self.end = end
        self.answer_spans = answer_spans

    @property
    def n_context_words(self):
        return sum(len(s) for s in self.text)


class ParagraphFilter(Configurable):
    def prune(self, question, paragraphs: List[ExtractedParagraph]):
        raise NotImplementedError()


class ContainsQuestionWord(ParagraphFilter):
    def __init__(self, stop, allow_first=True):
        self.stop = stop
        self.allow_first = allow_first

    def prune(self, question, paragraphs: List[ExtractedParagraph]):
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
        return output


class TopTfIdf(ParagraphFilter):
    def __init__(self, stop, n_to_select: int):
        self.stop = stop
        self.n_to_select = n_to_select

    def prune(self, question, paragraphs: List[ExtractedParagraph]):
        tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self.stop.words)
        text = []
        for para in paragraphs:
            text.append(" ".join(" ".join(s) for s in para.text))
        try:
            para_features = tfidf.fit_transform(text)
            q_features = tfidf.transform([" ".join(question)])
        except ValueError as e:
            print("TfIdf error: " + str(e))
            return []

        dists = pairwise_distances(q_features, para_features, "cosine").ravel()
        sorted_ix = np.argsort(dists)
        return [paragraphs[i] for i in sorted_ix[:self.n_to_select] if dists[i] > 0]


class DocumentSplitter(Configurable):

    @property
    def max_tokens(self):
        """ max number of tokens a paragraph from this splitter can have, or None """
        return None

    @property
    def reads_first_n(self):
        """ only requires the first `n` tokens of the documents, or None """
        return None

    def split(self, doc: List[List[List[str]]], spans: np.ndarray) -> List[ExtractedParagraph]:
        raise NotImplementedError()


class Truncate(DocumentSplitter):
    """ map a document to a single paragraph of the first `max_tokens` tokens """

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens

    def max_tokens(self):
        return self.max_tokens

    @property
    def reads_first_n(self):
        return self.max_tokens

    def split(self, doc: List[List[List[str]]], spans: np.ndarray):
        output = []
        cur_tokens = 0
        for para in doc:
            for sent in para:
                if cur_tokens + len(sent) > self.max_tokens:
                    output.append(sent[:self.max_tokens - cur_tokens])
                    return [ExtractedParagraph(output, 0, self.max_tokens, spans[spans[:, 1] < self.max_tokens])]
                else:
                    cur_tokens += len(sent)
                    output.append(sent)
        return [ExtractedParagraph(output, 0, cur_tokens, spans[spans[:, 1] < self.max_tokens])]


class MergeParagraphs(DocumentSplitter):
    PARAGRAPH_TOKEN = "%%PARAGRAPH%%"
    DOC_TOKEN = "%%DOCUMENT%%"

    """
    Build paragraphs that always start with document-paragraph, but might
    include other paragraphs. Paragraphs are always smaller then `max_tokens`
    (so paragraphs > `max_tokens` will always be truncated).
     """

    def __init__(self, max_tokens: int, top_n: int=None, pad=0):
        self.max_tokens = max_tokens
        self.top_n = top_n
        self.pad = pad

    def special_tokens(self):
        return []

    @property
    def reads_first_n(self):
        return self.top_n

    def max_tokens(self):
        return self.max_tokens

    def split(self, doc: List[List[List[str]]], spans: np.ndarray):
        all_paragraphs = []

        on_doc_token = 0  # the word in the document the current paragraph starts at
        on_paragraph = []  # text we have collect for the current paragraph
        cur_tokens = 0   # number of tokens in the current paragraph

        word_ix = 0
        for para in doc:
            n_words = sum(len(s) for s in para)
            if self.top_n is not None and (word_ix+self.top_n)>self.top_n:
                if word_ix == self.top_n:
                    break
                para = extract_tokens(para, self.top_n - word_ix)
                n_words = self.top_n - word_ix

            start_token = word_ix
            end_token = start_token + n_words
            word_ix = end_token

            if cur_tokens + n_words > self.max_tokens:
                if cur_tokens != 0:  # end the current paragraph
                    if self.pad > 0:
                        pad_with = min(self.max_tokens - cur_tokens, self.pad)
                        on_paragraph += extract_tokens(para, self.max_tokens - cur_tokens)
                        all_paragraphs.append(ExtractedParagraph(on_paragraph, on_doc_token, start_token+pad_with, None))
                    else:
                        all_paragraphs.append(ExtractedParagraph(on_paragraph, on_doc_token, start_token, None))
                    on_paragraph = []
                    cur_tokens = 0

                if n_words >= self.max_tokens:  # either truncate the given paragraph, or begin a new paragraph
                    text = extract_tokens(para, self.max_tokens)
                    all_paragraphs.append(ExtractedParagraph(text, start_token,
                                                             start_token+self.max_tokens, None))
                    on_doc_token = end_token
                else:
                    on_doc_token = start_token
                    on_paragraph += para
                    cur_tokens = n_words
            else:
                on_paragraph += para
                cur_tokens += n_words

        if len(on_paragraph) > 0:
            all_paragraphs.append(ExtractedParagraph(on_paragraph, on_doc_token, word_ix, None))

        for para in all_paragraphs:
            para.answer_spans = spans[np.logical_and(spans[:, 0] >= para.start, spans[:, 1] < para.end)] - para.start

        return all_paragraphs


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
        print(doc)
        # if doc != "web/98/98_1250432":
        #     continue
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

        paragraphs = splitter.split(text, fake_answers)

        for para in paragraphs:
            text = flatten_iterable(para.text)
            if max_tokens is not None and len(text) > max_tokens:
                raise ValueError("Paragraph len len %d, but max tokens was %d" % (len(text), max_tokens))
            plain = [x for x in text if x not in splitter.special_tokens()]
            start, end = para.start, para.end
            if plain != flattened[start:end]:
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
    # n, bins, patches = plt.hist(para_lens[para_lens < 1000], bins=100)
    # l = plt.plot(bins, n, 'r--', linewidth=1)
    # plt.show()


if __name__ == "__main__":
    test_splitter(MergeParagraphs(200, pad=True), 1000, 20, seed=0)
    # show_paragraph_lengths()




