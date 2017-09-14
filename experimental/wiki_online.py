import json
import time
from difflib import HtmlDiff
from os import mkdir
from os.path import exists, join
from typing import List, Dict

import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy import optimize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances

from config import CORPUS_DIR
from configurable import Configurable
from utils import flatten_iterable

RAW_CACHE = "/tmp/cache.json"  # TODO remove?
WIKI_API = "https://en.wikipedia.org/w/api.php"


def align_paragraphs(paragraphs1: List[List[str]], paragraphs2: List[List[str]]):
    # Simple bag-of-words approach seems to work fine
    cv = CountVectorizer(stop_words="english")
    cv.fit(list(" ".join(para) for para in paragraphs1) + list(" ".join(para) for para in paragraphs2))
    p1_features = cv.transform(list(" ".join(para) for para in paragraphs1))
    p2_features = cv.transform(list(" ".join(para) for para in paragraphs2))
    dist = pairwise_distances(p1_features, p2_features, metric="cosine")

    r, c = optimize.linear_sum_assignment(dist)
    alignmnet_costs = dist[r, c]
    return r, c, alignmnet_costs


def show_alignment_errors(para1, para2, r, c, keep):
    lines1 = []
    lines2 = []
    for i,j,k in zip(r,c,keep):
        if k:
            lines1.append(" ".join(para1[i]))
            lines2.append(" ".join(para2[j]))

    return HtmlDiff(tabsize=4, wrapcolumn=120).make_file(lines1, lines2)


def show_complete_alignment(para1, para2, r, c):
    lines1 = []
    lines2 = []
    for i, para in enumerate(para1):
        ix = np.where(r == i)[0]
        if len(ix) == 0:
            lines2.append("NONE")
        else:
            lines2.append(" ".join(para2[c[ix[0]]]))
        lines1.append(" ".join(para))

    return HtmlDiff(tabsize=4, wrapcolumn=120).make_file(lines1, lines2)


# class TimestampedWikiCorpus(Configurable):
#     DOC_DIR_NAME = "docs"
#     TOKENIZED_DIR_NAME = "tokenized"
#
#     def __init__(self, timestamp, tokenizer="NLTK_AND_CLEAN"):
#         self.timestamp = timestamp
#         self.tokenizer = tokenizer
#         name = join(CORPUS_DIR, "wiki-" + str(self.timestamp))
#         if not exists(name):
#             mkdir(name)
#         self.doc_cache = join(name, self.DOC_DIR_NAME)
#         self.tokenized_cache = join(name, self.TOKENIZED_DIR_NAME + "-" + self.tokenizer)
#         if not exists(self.doc_cache):
#             mkdir(self.doc_cache)
#             mkdir(self.tokenized_cache)
#
#     def _get_tokenized_filename(self, title):
#         return join(self.tokenized_cache, title.replace(" ", "_") + ".json")
#
#     def _get_raw_filename(self, title):
#         return join(self.doc_cache, title.replace(" ", "_") + ".json")
#
#     def get_wiki_article(self, wiki_title) -> WikiArticle:
#         tokenized_file = self._get_tokenized_filename(wiki_title)
#         if exists(tokenized_file):
#             with open(tokenized_file, "r") as f:
#                 data = json.load(f)
#                 return WikiArticle(data["title"], [WikiParagraph(**x) for x in data["paragraphs"]])
#
#         print("Parsing and Tokenizing " + wiki_title)
#         raw_data = self._get_raw_article(wiki_title)["parse"]
#
#         sent_tokenize, word_tokenize = get_paragraph_tokenizer(self.tokenizer)
#
#         soup = BeautifulSoup(raw_data["text"]["*"], "lxml")
#         paragraphs = []
#         sections = []
#         for x in soup.findAll(["p", "h2", "h3", "h4", "h5", "h6"]):
#             if x.name[0] == "h":
#                 if x.get_text() == "Contents":
#                     continue
#                 sect_name = x.find(attrs={"class": "mw-headline"}).get_text()
#                 level = int(x.name[1:]) - 1
#                 if len(sections) < level - 1:
#                     print("Warning, mismatch sections")
#                     while len(sections) < level - 1:
#                         sections.append("")
#                 sections = sections[:level]
#                 sections.append(sect_name)
#             else:
#                 for citation in x.findAll("sup", {"class": "reference"}):
#                     citation.extract()
#                 for sub in x.findAll("sup"):
#                     citations = sub.findAll("a", href=True)
#                     if len(citations) == 1:
#                         citation = citations[0]
#                         href = citation["href"]
#                         if href.startswith("#cite") or href == "/wiki/Wikipedia:Citation_needed":
#                             sub.extract()
#                 txt = x.get_text()
#
#                 text = [word_tokenize(sent) for sent in sent_tokenize(txt)]
#
#                 para = WikiParagraph(len(paragraphs), sections, text)
#                 paragraphs.append(para)
#
#         article = WikiArticle(wiki_title, paragraphs)
#         with open(tokenized_file, "w") as f:
#             json.dump(article, f, default=lambda x: x.__dict__)
#         return article
#
#     def _get_raw_article(self, wiki_title) -> Dict:
#         raw_file = self._get_raw_filename(wiki_title)
#         if exists(raw_file):
#             with open(raw_file, "r") as f:
#                 return json.load(f)
#
#         print("Fetching " + wiki_title)
#         r = requests.get(WIKI_API,
#                          params=dict(action="query", prop="revisions",
#                                      rvstart=self.timestamp,
#                                      rvlimit=1, format="json",
#                                      redirects=True, titles=wiki_title))
#         data = r.json()
#         query = data["query"]
#         if "redirects" in query:
#             re = query["redirects"]
#             if len(re) != 1:
#                 raise RuntimeError()
#             print("Title <%s> redirected to: <%s>" % (re[0]["from"], re[0]["to"]))
#
#         pages = query["pages"]
#         if len(pages) != 1:
#             raise RuntimeError()
#         page = next(iter(pages.values()))
#
#         revid = page["revisions"][0]["parentid"]
#
#         r = requests.get(WIKI_API, params=dict(action="parse", oldid=revid, format="json"))
#         data = r.json()
#
#         with open(raw_file, "w") as f:
#             json.dump(data, f)
#         return data



class SquadWikiArticles(WikiApiCorpus):
    def __init__(self):
        # My best guess at the timestamp to get articles for SQuAD, it seems to work pretty well
        super().__init__(1457078400, "NLTK_AND_CLEAN")


def pre_load_squad_wiki():
    """ Pre-cache the SQuAD articles  """
    data = SquadCorpus()
    docs = data.get_train_docs()
    docs += data.get_dev_docs()
    corpus = SquadWikiArticles()
    for i,squad_article in enumerate(docs):
        print("Fetching article %s (%d/%d)" % (squad_article.wiki_title, i+1, len(docs)))
        corpus.get_wiki_article(squad_article.wiki_title)
        time.sleep(1)


def show_squad_alignments(out_dir, thresh=0.1, wrong_paragraphs_only: bool=False):
    """ Show alignemnts between our articles and the SQuAD articles  """
    data = SquadCorpus()
    docs = data.get_train_docs()
    docs += data.get_dev_docs()
    corpus = SquadWikiArticles()

    costs = []
    for i,squad_article in enumerate(docs):
        squad_para = [flatten_iterable(x.context) for x in squad_article.paragraphs]
        wiki_article = corpus.get_wiki_article(squad_article.wiki_title)
        wiki_para = [flatten_iterable(x.text) for x in wiki_article.paragraphs]

        r,c,cost = align_paragraphs(wiki_para, squad_para)
        costs.append(cost)
        max_cost = np.max(cost)
        if max_cost > thresh:
            print(wiki_article.title + "  " + str(max_cost))
            if wrong_paragraphs_only:
                diff = show_alignment_errors(wiki_para, squad_para, r, c, cost > thresh)
            else:
                diff = show_complete_alignment(wiki_para, squad_para, r, c)

            with open(join(out_dir, wiki_article.title.replace(" ", "_") +
                    ("-%.4f"%max_cost) + ".html"), "w") as f:
                f.write(diff)

    print("Mean max cost: %%.5f" % (np.mean([x.max() for x in costs])))
    print("Mean cost: %.5f" % (np.mean([x.mean() for x in costs])))


if __name__ == "__main__":
    show_squad_alignments("/tmp/alignment_errors")