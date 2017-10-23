from os.path import join
from typing import Dict, List

from docqa.config import DOCUMENT_READER_DB, CORPUS_DIR
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, ParagraphWithInverse
from docqa.squad.build_squad_dataset import clean_title
from docqa.squad.squad_data import SquadCorpus, Document
import sqlite3


"""
Retrive documents by title from a sqlite database, we used this to evaluate our
model on the documents from https://github.com/facebookresearch/DrQA
"""


def build_corpus_subset(output):
    docs = SquadCorpus().get_dev()
    titles = [clean_title(doc.title) for doc in docs]
    for i, t in enumerate(titles):
        if t == "Sky (United Kingdom)":
            titles[i] = "Sky UK"

    with sqlite3.connect(DOCUMENT_READER_DB) as conn:
        c = conn.cursor()

        c.execute("CREATE TEMPORARY TABLE squad_docs(id)")
        c.executemany("INSERT INTO squad_docs VALUES (?)", [(x,) for x in titles])

        c.execute("ATTACH DATABASE ? AS db2", (output, ))
        c.execute("CREATE TABLE db2.documents (id PRIMARY KEY, text);")

        c.execute("INSERT INTO db2.documents SELECT * FROM documents WHERE id in squad_docs")
        c.close()


def get_doc_rd_doc(docs: List[Document]) -> Dict[str, List[ParagraphWithInverse]]:
    tokenizer = NltkAndPunctTokenizer()
    conn = sqlite3.connect(DOCUMENT_READER_DB)
    c = conn.cursor()
    titles = [clean_title(doc.title) for doc in docs]
    for i, t in enumerate(titles):
        # Had to manually resolve this (due to changes in Wikipedia?)
        if t == "Sky (United Kingdom)":
            titles[i] = "Sky UK"

    title_to_doc_id = {t: doc.title for t, doc in zip(titles, docs)}

    c.execute("CREATE TEMPORARY TABLE squad_docs(id)")
    c.executemany("INSERT INTO squad_docs VALUES (?)", [(x,) for x in titles])

    c.execute("SELECT id, text FROM documents WHERE id IN squad_docs")

    documents = {}
    out = c.fetchall()
    conn.close()
    for title, text in out:
        paragraphs = []
        for para in text.split("\n"):
            para = para.strip()
            if len(para) > 0:
                paragraphs.append(tokenizer.tokenize_with_inverse(para))
        documents[title_to_doc_id[title]] = paragraphs

    return documents

if __name__ == "__main__":
    build_corpus_subset(join(CORPUS_DIR, "doc-rd-subset.db"))
