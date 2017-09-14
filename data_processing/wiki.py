import json
import unicodedata
from os import mkdir
from os.path import join, exists
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from configurable import Configurable
from data_processing.text_utils import NltkAndPunctTokenizer, ParagraphWithInverse

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


class WikiCorpus(Configurable):
    """
    Class the can download wiki-articles and return them as tokenized text
    """

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

        # Wiki html is pretty structured, so this seems to work reasonable well
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
                # remove citations
                for citation in element.findAll("sup", {"class": "reference"}):
                    citation.extract()

                # remove citation needed
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