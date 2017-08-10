from typing import List

import numpy as np

from data_processing.paragraph_qa import Document, Paragraph
from data_processing.qa_data import Answer
from data_processing.wiki_online import WikiParagraph, WikiApiCorpus, align_paragraphs, SquadWikiArticles
from squad.squad import SquadCorpus
from utils import flatten_iterable

"""
QA using the original wiki article (instead of just the paragraphs used in SQaUD)
"""

class DocumentQuestion(object):
    def __init__(self, words: List[str], question_id: str, paragraph_num: int, answer: Answer):
        self.words = words
        self.question_id = question_id
        self.paragraph_num = paragraph_num
        self.answer = answer


class WikiQaDocument(object):
    def __init__(self, paragraphs: List[WikiParagraph], questions: List[DocumentQuestion]):
        self.paragraphs = paragraphs
        self.questions = questions


class WikiArticleQaCorpus(object):

    def __init__(self, squad_corpus, wiki_source: WikiApiCorpus, swap_paragraphs: bool,
                 alignemnt_thresh = 0.2):
        self.squad_corpus = squad_corpus
        self.wiki_source = wiki_source
        self.alignemnt_thresh = alignemnt_thresh
        self.swap_paragraphs = swap_paragraphs

    def get_train_docs(self):
        return self.align_docs(self.squad_corpus.get_train_docs(), False)

    def get_dev_docs(self):
        return self.align_docs(self.squad_corpus.get_dev_docs(), False)

    def align_docs(self, docs: List[Document], log=False) -> List[Document]:
        aligned = []
        for doc in docs:
            wiki_article = self.wiki_source.get_wiki_article(doc.wiki_title)

            # FIXME, this should never occur
            wiki_article.paragraphs = [x for x in wiki_article.paragraphs if len(x.text) > 0]

            wiki_para = [flatten_iterable(x.text) for x in wiki_article.paragraphs]
            squad_para = [flatten_iterable(x.context) for x in doc.paragraphs]

            alignment_r, alignment_c, cost = align_paragraphs(wiki_para, squad_para)

            paragraphs = []

            # for each wiki paragraph, if it has a strongly aligned squad paragraph,
            # and that paragrahp's question in and optionally swap in that paragraph
            used_paragraph = []
            for i in range(len(wiki_para)):
                ix = np.where(alignment_r == i)[0]
                if len(ix) == 0:  # not aligned, include with no questions
                    paragraphs.append(Paragraph(wiki_article.paragraphs[i].text,
                                                [], wiki_article.title, i))
                else:
                    score = cost[ix[0]]
                    if score < self.alignemnt_thresh:
                        used_paragraph.append(alignment_c[ix[0]])
                        aligned_paragraph = doc.paragraphs[alignment_c[ix[0]]]
                        # Use the squad's paragraphs questions, and possibly its text
                        if self.swap_paragraphs:
                            text = aligned_paragraph.context
                        else:
                            text = wiki_article.paragraphs[i].text
                        paragraphs.append(Paragraph(text, aligned_paragraph.questions,
                                                    wiki_article.title, i))
                    else:  # else use the plain wiki paragraph w/o questions
                        paragraphs.append(Paragraph(wiki_article.paragraphs[i].text,
                                                    [], wiki_article.title, i))

            skipped_paragraphs = set(range(0, len(doc.paragraphs))).difference(used_paragraph)
            if len(skipped_paragraphs) > 0 and log:
                print("Skipped %d paragraphs from doc %s" % (len(skipped_paragraphs), doc.title))

            aligned.append(Document(doc.doc_id, doc.title, wiki_article.title, paragraphs))

        return aligned


def main():
    corp = WikiArticleQaCorpus(SquadCorpus(), SquadWikiArticles(), True, 0.15)
    corp.get_train_docs()

if __name__ == "__main__":
    main()

