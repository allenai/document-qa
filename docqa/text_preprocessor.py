from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from docqa.utils import flatten_iterable

from docqa.data_processing.document_splitter import ExtractedParagraphWithAnswers, MergeParagraphs, ExtractedParagraph
from docqa.data_processing.multi_paragraph_qa import ParagraphWithAnswers
from docqa.configurable import Configurable
from docqa.squad.squad_data import SquadCorpus
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset


class TextPreprocessor(Configurable):
    """ Preprocess text input, must be deterministic. Only used thus far adding special indicator tokens """

    def encode_extracted_paragraph(self, question: List[str], paragraph: ExtractedParagraphWithAnswers):
        text, answers, _ = self.encode_paragraph(question, paragraph.text,
                                                 paragraph.start == 0, paragraph.answer_spans)
        return ParagraphWithAnswers(text, answers)

    def encode_text(self, question: List[str], paragraph: ExtractedParagraph):
        text, _, _ = self.encode_paragraph(question, paragraph.text, paragraph.start == 0,
                                           np.zeros((0, 2), dtype=np.int32))
        return text

    def encode_paragraph(self, question: List[str], paragraphs: List[List[str]],
                         is_first, answer_spans: np.ndarray,
                         token_spans=None) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
        """
        Returns updated (and flattened) text, answer_spans, and token_spans
        """
        raise NotImplementedError()

    def special_tokens(self) -> List[str]:
        return []


class WithIndicators(TextPreprocessor):
    """
    Adds a document or group start token before the text, and a paragraph token between each
    between in each paragraph.
    """

    PARAGRAPH_TOKEN = "%%PARAGRAPH%%"
    DOCUMENT_START_TOKEN = "%%DOCUMENT%%"
    PARAGRAPH_GROUP = "%%PARAGRAPH_GROUP%%"

    def __init__(self, remove_cross_answer: bool=True, para_tokens: bool=True, doc_start_token: bool=True):
        self.remove_cross_answer = remove_cross_answer
        self.doc_start_token = doc_start_token
        self.para_tokens = para_tokens

    def special_tokens(self) -> List[str]:
        tokens = [self.PARAGRAPH_GROUP]
        if self.doc_start_token:
            tokens.append(self.DOCUMENT_START_TOKEN)
        if self.para_tokens:
            tokens.append(self.PARAGRAPH_TOKEN)
        return tokens

    def encode_paragraph(self, question: List[str], paragraphs: List[List[str]], is_first, answer_spans: np.ndarray, inver=None):
        out = []

        offset = 0
        if self.doc_start_token and is_first:
            out.append(self.DOCUMENT_START_TOKEN)
        else:
            out.append(self.PARAGRAPH_GROUP)

        if inver is not None:
            inv_out = [np.zeros((1, 2), dtype=np.int32)]
        else:
            inv_out = None

        offset += 1
        spans = answer_spans + offset

        out += paragraphs[0]
        offset += len(paragraphs[0])
        on_ix = len(paragraphs[0])
        if inv_out is not None:
            inv_out.append(inver[:len(paragraphs[0])])

        for sent in paragraphs[1:]:
            if self.remove_cross_answer:
                remove = np.logical_and(spans[:, 0] < offset, spans[:, 1] >= offset)
                spans = spans[np.logical_not(remove)]

            if self.para_tokens:
                spans[spans[:, 0] >= offset, 0] += 1
                spans[spans[:, 1] >= offset, 1] += 1

                out.append(self.PARAGRAPH_TOKEN)
                if inv_out is not None:
                    if len(inv_out) == 0 or len(inv_out[-1]) == 0:
                        inv_out.append(np.zeros((1, 2), dtype=np.int32))
                    else:
                        inv_out.append(np.full((1, 2), inv_out[-1][-1][1], dtype=np.int32))
                offset += 1

            out += sent
            offset += len(sent)
            if inv_out is not None:
                inv_out.append(inver[on_ix:on_ix+len(sent)])
            on_ix += len(sent)

        return out, spans, None if inv_out is None else np.concatenate(inv_out)

    def __setstate__(self, state):
        if "state" in state:
            state["state"]["doc_start_token"] = True
            state["state"]["para_tokens"] = True
        else:
            if "doc_start_token" not in state:
                state["doc_start_token"] = True
            if "para_tokens" not in state:
                state["para_tokens"] = True
        super().__setstate__(state)


def check_preprocess():
    data = TriviaQaWebDataset()
    merge = MergeParagraphs(400)
    questions = data.get_dev()
    pre = WithIndicators(False)
    remove_cross = WithIndicators(True)
    rng = np.random.RandomState(0)
    rng.shuffle(questions)

    for q in tqdm(questions[:1000]):
        doc = rng.choice(q.all_docs, 1)[0]
        text = data.evidence.get_document(doc.doc_id, n_tokens=800)
        paras = merge.split_annotated(text, doc.answer_spans)
        para = paras[np.random.randint(0, len(paras))]
        built = pre.encode_extracted_paragraph(q.question, para)

        expected_text = flatten_iterable(para.text)
        if expected_text != [x for x in built.text if x not in pre.special_tokens()]:
            raise ValueError()

        expected = [expected_text[s:e+1] for s, e in para.answer_spans]
        expected = Counter([tuple(x) for x in expected])

        actual = [tuple(built.text[s:e+1]) for s,e in built.answer_spans]
        actual_cleaned = Counter(tuple(z for z in x if z not in pre.special_tokens()) for x in actual)
        if actual_cleaned != expected:
            raise ValueError()

        r_built = remove_cross.encode_extracted_paragraph(q.question, para)
        rc = Counter(tuple(r_built.text[s:e + 1]) for s, e in r_built.answer_spans)
        removed = Counter()
        for w in actual:
            if all(x not in pre.special_tokens() for x in w):
                removed[w] += 1

        if rc != removed:
            raise ValueError()


def check_preprocess_squad():
    data = SquadCorpus().get_train()
    remove_cross = WithIndicators(True)

    for doc in tqdm(data):
        for para in doc.paragraphs:
            q = para.questions[np.random.randint(0, len(para.questions))]

            text, ans, inv = remove_cross.encode_paragraph(q.words, para.text, para.paragraph_num == 0,
                                       q.answer.answer_spans, para.spans)
            if len(inv) != len(text):
                raise ValueError()
            for i in range(len(inv)-1):
                if inv[i, 0] > inv[i+1, 0]:
                    raise ValueError()
            for (s1, e1), (s2, e2) in zip(ans, q.answer.answer_spans):
                if tuple(inv[s1]) != tuple(para.spans[s2]):
                    raise ValueError()
                if tuple(inv[e1]) != tuple(para.spans[e2]):
                    raise ValueError()


if __name__ == "__main__":
    check_preprocess_squad()