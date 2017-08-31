import json
from typing import List, Dict, Optional

import requests
from os.path import join, exists

from os import mkdir

import unicodedata
from bs4 import BeautifulSoup

from config import CORPUS_DIR
from configurable import Configurable
from data_processing.text_utils import NltkAndPunctTokenizer, ParagraphWithInverse
from utils import flatten_iterable
import numpy as np

WIKI_API = "https://en.wikipedia.org/w/api.php"


def get_wiki_page_ids(article_titles, per_query=25):
    """
    Utility method to get the page_id and resolve re-directs for a set of wikipedia titles
    """
    wiki_page_ids = {}

    for i in range((len(article_titles) + per_query - 1) // per_query):
        start = i * per_query
        end = min((i + 1) * per_query, len(article_titles))
        original_titles = article_titles[start:end]

        r = requests.get(WIKI_API,
                         params=dict(action="query", format="json",
                                     redirects=True,
                                     titles="|".join(original_titles)))
        data = r.json()

        query = data["query"]
        if "redirects" in query:
            redirects = {x["to"]: x["from"] for x in query["redirects"]}
        else:
            redirects = {}

        for page, page_data in query["pages"].items():
            page = int(page)
            title = page_data["title"]

            if page == -1:
                raise ValueError()

            original_title = redirects.get(title, title)
            if original_title not in original_titles:
                raise ValueError(title)

            wiki_page_ids[original_title] = (title, page)

    return [wiki_page_ids[x] for x in article_titles]


class WikiParagraph(ParagraphWithInverse):
    def __init__(self, paragraph_num: int, kind: str, text: List[List[str]],
                 original_text: Optional[str]=None, span_mapping=None):
        super().__init__(text, original_text, span_mapping)
        self.paragraph_num = paragraph_num
        self.kind = kind


class WikiArticle(object):
    def __init__(self, title: str, page_id: int, paragraphs: List[WikiParagraph]):
        self.title = title
        self.page_id = page_id
        self.paragraphs = paragraphs

    @property
    def url(self):
        return "https://en.wikipedia.org/?curid=" + str(self.page_id)


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


class WikiCorpus(Configurable):

    def __init__(self, cache_dir=None, follow_redirects: bool=True,
                 keep_inverse_mapping: bool=False,
                 extract_lists: bool=False, tokenizer=NltkAndPunctTokenizer()):
        self.tokenizer = tokenizer
        self.extract_lists = extract_lists
        self.follow_redirects = follow_redirects
        self.cache_dir = cache_dir
        self.keep_inverse_mapping = keep_inverse_mapping

        if cache_dir is not None and not exists(self.cache_dir):
            mkdir(self.cache_dir)

    def _get_tokenized_filename(self, title):
        title = unicodedata.normalize('NFKC', title).lower()
        return join(self.cache_dir, title.replace(" ", "_")
                    .replace("/", "-") + ".json")

    def _text_to_paragraph(self, ix, kind: str, text: str):
        if not self.keep_inverse_mapping:
            text = self.tokenizer.tokenize_paragraph(text)
            return WikiParagraph(ix, kind, text)
        else:
            para = self.tokenizer.tokenize_with_inverse(text)
            return WikiParagraph(ix, kind, para.text, para.original_text, para.spans)

    def _sent_to_paragraph(self, ix, kind: str, text: List[str]):
        if not self.keep_inverse_mapping:
            tokenized = [self.tokenizer.tokenize_sentence(s) for s in text]
            return WikiParagraph(ix, kind, tokenized)
        else:
            para = ParagraphWithInverse.concat(
                [self.tokenizer.tokenize_with_inverse(x, True) for x in text], " ")
            return WikiParagraph(ix, kind, para.text, para.original_text, para.spans)

    def get_wiki_article(self, wiki_title) -> WikiArticle:
        if self.cache_dir is not None:
            tokenized_file = self._get_tokenized_filename(wiki_title)
            if exists(tokenized_file):
                with open(tokenized_file, "r") as f:
                    data = json.load(f)
                    return WikiArticle(data["title"], data["url"], [WikiParagraph(**x) for x in data["paragraphs"]])

        r = requests.get(WIKI_API, dict(action="parse", page=wiki_title,
                                        redirects=self.follow_redirects, format="json"))

        if r.status_code != 200:
            raise ValueError()
        raw_data = r.json()["parse"]

        soup = BeautifulSoup(raw_data["text"]["*"], "lxml")
        paragraphs = []
        to_find = ["p", "h2", "h3", "h4", "h5", "h6"]
        if self.extract_lists:
            to_find += ["ul", "ol"]
        for element in soup.findAll(to_find):
            if element.name[0] == "h":
                if element.get_text() == "Contents":
                    continue
                sect_name = element.find(attrs={"class": "mw-headline"}).get_text()
                para = self._sent_to_paragraph(len(paragraphs), "section", [sect_name])
                if para.n_tokens > 0:
                    paragraphs.append(para)
            elif element.name == "ul" or element.name == "ol":
                if dict(element.parent.attrs).get("class") != ["mw-parser-output"]:
                    # only extract "body" lists
                    continue
                para = self._sent_to_paragraph(len(paragraphs),
                                               "list" if element.name == "ul" else "ordered_list",
                                               [x.get_text() for x in element.findAll("li")])
                if para.n_tokens > 0:
                    paragraphs.append(para)
            else:
                for citation in element.findAll("sup", {"class": "reference"}):
                    citation.extract()
                for sub in element.findAll("sup"):
                    citations = sub.findAll("a", href=True)
                    if len(citations) == 1:
                        citation = citations[0]
                        href = citation["href"]
                        if href.startswith("#cite") or href == "/wiki/Wikipedia:Citation_needed":
                            sub.extract()
                text = element.get_text()
                para = self._text_to_paragraph(len(paragraphs), "paragraph", text)
                if para.n_tokens > 0:
                    paragraphs.append(para)

        article = WikiArticle(wiki_title, raw_data["pageid"], paragraphs)
        if self.cache_dir is not None:
            with open(tokenized_file, "w") as f:
                json.dump(article, f, default=lambda x: x.__dict__)
        return article


if __name__ == "__main__":
    from data_processing.document_splitter import MergeParagraphs
    doc = WikiCorpus(keep_inverse_mapping=True).get_wiki_article("Queen Elizabeth 2")
    MergeParagraphs(400).split_inverse(doc.paragraphs)
    pass