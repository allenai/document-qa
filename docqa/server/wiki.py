import logging
import ujson
import unicodedata
from os import mkdir
from os.path import exists, join
from typing import Optional, List

import numpy as np
from aiohttp import ClientSession
from bs4 import BeautifulSoup

from docqa.data_processing.text_utils import NltkAndPunctTokenizer, ParagraphWithInverse

WIKI_API = "https://en.wikipedia.org/w/api.php"

log = logging.getLogger('wiki')


class WikiParagraph(ParagraphWithInverse):
    def __init__(self, paragraph_num: int, kind: str, text: List[List[str]],
                 original_text: Optional[str]=None, span_mapping=None):
        """
        :param paragraph_num: Paragraph number within the overall document
        :param kind: Kind of paragraph (eg "title", "list", "ordered-list", "section")
        :param text: Sentences and words
        :param original_text: Optional, original text
        :param span_mapping: Optional, mapping of tokens -> span within the original text
        """
        super().__init__(text, original_text, span_mapping)
        self.paragraph_num = paragraph_num
        self.kind = kind

    def to_json(self):
        if self.spans is not None:
            word_ix = 0
            compact = []
            for sent in self.text:
                tokens = []
                for word in sent:
                    s, e = self.spans[word_ix]
                    s, e = int(s), int(e)  # ujson doesn't play nice with numpy
                    if word == self.original_text[s:e]:
                        tokens.append((s, e))
                    else:
                        tokens.append((word,  s, e))
                    word_ix += 1
                compact.append(tokens)
            return dict(paragraph_num=self.paragraph_num,
                        kind=self.kind,
                        original_text=self.original_text,
                        spans=compact)
        else:
            return self.__dict__

    @staticmethod
    def from_json(data):
        if data["spans"] is not None:
            original_text = data["original_text"]
            spans = []
            text = []
            for sent in data["spans"]:
                sent_tokens = []
                for tup in sent:
                    if len(tup) == 2:
                        spans.append(tup)
                        sent_tokens.append(original_text[tup[0]:tup[1]])
                    else:
                        spans.append(tup[1:])
                        sent_tokens.append(tup[0])
                text.append(sent_tokens)
            return WikiParagraph(data["paragraph_num"], data["kind"], text, original_text,
                                 np.array(spans, dtype=np.int32))
        else:
            return WikiParagraph(**data)


class WikiArticle(object):
    def __init__(self, title: str, page_id: int, paragraphs: List[WikiParagraph]):
        """
        :param title: Article's title
        :param page_id: wiki page id
        :param paragraphs: paragraphs within the document
        """
        self.title = title
        self.page_id = page_id
        self.paragraphs = paragraphs

    @property
    def url(self):
        return "https://en.wikipedia.org/?curid=" + str(self.page_id)


class WikiCorpus(object):
    """
    Class the can download wiki-articles and return them as tokenized text
    """

    def __init__(self, cache_dir=None, follow_redirects: bool=True,
                 keep_inverse_mapping: bool=False,
                 extract_lists: bool=False, tokenizer=NltkAndPunctTokenizer(),
                 loop=None):
        """
        :param cache_dir: Optional, directory to cache the documents we download
        :param follow_redirects: Follow wiki re-directs
        :param keep_inverse_mapping: Keep track of the inverse mapping of tokens so the text can
                                     be "untokenized" accurately
        :param extract_lists: Include lists in the extracted articles
        :param tokenizer: Tokenizer to use to tokenize the documents
        """
        self.cl_sess = ClientSession(loop=loop)
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
        text = text.strip()
        if not self.keep_inverse_mapping:
            text = self.tokenizer.tokenize_paragraph(text)
            return WikiParagraph(ix, kind, text)
        else:
            para = self.tokenizer.tokenize_with_inverse(text)
            return WikiParagraph(ix, kind, para.text, para.original_text, para.spans)

    def _sent_to_paragraph(self, ix, kind: str, text: List[str]):
        if not self.keep_inverse_mapping:
            tokenized = [self.tokenizer.tokenize_sentence(s.strip()) for s in text]
            return WikiParagraph(ix, kind, tokenized)
        else:
            para = ParagraphWithInverse.concat(
                [self.tokenizer.tokenize_with_inverse(x.strip(), True) for x in text], " ")
            return WikiParagraph(ix, kind, para.text, para.original_text, para.spans)

    async def get_wiki_article(self, wiki_title: str) -> WikiArticle:
        # Note client is responsible for rate limiting as needed
        if self.cache_dir is not None:
            tokenized_file = self._get_tokenized_filename(wiki_title)
            if exists(tokenized_file):
                log.info("Load wiki article for \"%s\" from cache", wiki_title)
                with open(tokenized_file, "r") as f:
                    data = ujson.load(f)
                    return WikiArticle(data["title"], data["url"], [WikiParagraph.from_json(x) for
                                                                    x in data["paragraphs"]])

        log.info("Load wiki article for \"%s\"", wiki_title)

        # Use int(self.follow_redirects) since this get method doesn't support
        # bool values for some reason
        async with self.cl_sess.get(url=WIKI_API,
                                    params=dict(action="parse", page=wiki_title,
                                                redirects=int(self.follow_redirects),
                                                format="json")) as resp:
            data = await resp.json()

        raw_data = data["parse"]

        # Extract paragraph based on HTML tags
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
                    # only extract "body" lists, avoid info-boxes
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
            # save to cache
            with open(tokenized_file, "w") as f:
                ujson.dump(dict(title=article.title, url=article.url,
                                paragraphs=[x.to_json() for x in article.paragraphs]), f)
        return article

    def close(self):
        self.cl_sess.close()
