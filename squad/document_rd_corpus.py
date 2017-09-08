from os.path import join
from typing import Dict, List

from config import DOCUMENT_READER_DB, CORPUS_DIR
from data_processing.text_utils import NltkAndPunctTokenizer, ParagraphWithInverse
from squad.build_squad_dataset import clean_title
from squad.squad_data import SquadCorpus
import sqlite3


class DocumentReaderCorpus():
    pass


def build_corpus_subset(output):
    docs = SquadCorpus().get_dev()
    titles = [clean_title(doc.title) for doc in docs]
    for i, t in enumerate(titles):
        if t == "Sky (United Kingdom)":
            titles[i] = "Sky UK"

    conn = sqlite3.connect(DOCUMENT_READER_DB)
    c = conn.cursor()

    c.execute("CREATE TEMPORARY TABLE squad_docs(id)")
    c.executemany("INSERT INTO squad_docs VALUES (?)", [(x,) for x in titles])

    c.execute("ATTACH DATABASE ? AS db2", (output, ))
    c.execute("CREATE TABLE db2.documents (id PRIMARY KEY, text);")

    c.execute("INSERT INTO db2.documents SELECT * FROM documents WHERE id in squad_docs")
    c.close()


def get_doc_rd_doc(docs) -> Dict[str, List[ParagraphWithInverse]]:
    tokenizer = NltkAndPunctTokenizer()
    conn = sqlite3.connect(DOCUMENT_READER_DB)
    c = conn.cursor()
    titles = [clean_title(doc.title) for doc in docs]
    for i, t in enumerate(titles):
        if t == "Sky (United Kingdom)":
            titles[i] = "Sky UK"

    title_to_doc_id = {t: doc.title for t, doc in zip(titles, docs)}

    c.execute("CREATE TEMPORARY TABLE squad_docs(id)")
    c.executemany("INSERT INTO squad_docs VALUES (?)", [(x,) for x in titles])

    c.execute("SELECT id FROM squad_docs")

    c.execute("SELECT id, text FROM documents WHERE id IN squad_docs")

    docuemnts = {}
    out = c.fetchall()
    conn.close()
    for title, text in out:
        paragraphs = [tokenizer.tokenize_with_inverse(x) for x in text.split(" ")]
        docuemnts[title_to_doc_id[title]] = paragraphs

    return docuemnts

if __name__ == "__main__":
    build_corpus_subset(join(CORPUS_DIR, "doc-rd-subset.db"))
