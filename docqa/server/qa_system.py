import logging
import re
import time
from typing import Union, Optional, List, Tuple, Set

import numpy as np
import tensorflow as tf
from aiohttp import ClientSession

from docqa.data_processing.document_splitter import DocumentSplitter, ParagraphFilter
from docqa.data_processing.qa_training_data import ParagraphAndQuestionSpec, ParagraphAndQuestion
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, ParagraphWithInverse
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.model_dir import ModelDir
from docqa.server.web_searcher import AsyncWebSearcher, AsyncBoilerpipeCliExtractor
from docqa.server.wiki import WikiCorpus
from docqa.utils import ResourceLoader

TAGME_API = "https://tagme.d4science.org/tagme/tag"


class WebParagraph(ParagraphWithInverse):
    """ Paragraph the includes a url and source """

    def __init__(self, text: List[List[str]], original_text: str, token_spans: np.ndarray,
                 paragraph_num: int, start: int, end: int, source_name, source_url):
        super().__init__(text, original_text, token_spans)
        self.source_name = source_name
        self.paragraph_num = paragraph_num
        self.source_url = source_url
        self.start = start
        self.end = end


class QaSystem(object):
    """
    End-to-end QA system, uses web-requests to get relevant documents and a model
    to score candidate answer spans.
    """
    # TODO fix logging level

    _split_regex = re.compile("\s*\n\s*")  # split includes whitespace to avoid empty paragraphs

    def __init__(self,
                 wiki_cache: str,
                 paragraph_splitter: DocumentSplitter,
                 paragraph_selector: ParagraphFilter,
                 vocab: Union[str, None, Set[str]],
                 model: Union[ParagraphQuestionModel, ModelDir],
                 loader: ResourceLoader=ResourceLoader(),
                 bing_api_key=None,
                 bing_version="v5.0",
                 tagme_api_key=None,
                 blacklist_trivia_sites: bool=False,
                 n_dl_threads: int=5,
                 span_bound: int=8,
                 tagme_threshold: Optional[float]=0.2,
                 download_timeout: int=None,
                 n_web_docs=10,
                 loop=None):
        self.log = logging.getLogger('qa_system')
        self.tagme_threshold = tagme_threshold
        self.n_web_docs = n_web_docs
        self.blacklist_trivia_sites = blacklist_trivia_sites
        self.tagme_api_key = tagme_api_key

        self.client_sess = ClientSession(loop=loop)

        if bing_api_key is not None:
            if bing_version is None:
                raise ValueError("Must specify a Bing version if using a bing_api key")
            self.searcher = AsyncWebSearcher(bing_api_key, bing_version, loop=loop)
            self.text_extractor = AsyncBoilerpipeCliExtractor(n_dl_threads, download_timeout)
        else:
            self.text_extractor = None
            self.searcher = None

        if self.tagme_threshold is not None:
            self.wiki_corpus = WikiCorpus(wiki_cache, keep_inverse_mapping=True, loop=loop)
        else:
            self.wiki_corpus = None

        self.paragraph_splitter = paragraph_splitter
        self.paragraph_selector = paragraph_selector
        self.model_dir = model

        voc = None
        if vocab is not None:
            if isinstance(vocab, str):
                voc = set()
                with open(vocab, "r") as f:
                    for line in f:
                        voc.add(line.strip())
            else:
                voc = vocab
            self.log.info("Using preset vocab of size %d", len(voc))

        self.log.info("Setting up model...")
        if isinstance(model, ModelDir):
            self.model = model.get_model()
        else:
            self.model = model

        self.model.set_input_spec(ParagraphAndQuestionSpec(None), voc, loader)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with self.sess.as_default():
            pred = self.model.get_prediction()

        if isinstance(model, ModelDir):
            model.restore_checkpoint(self.sess)

        self.span_scores = pred.get_span_scores()
        self.span, self.score = pred.get_best_span(span_bound)
        self.tokenizer = NltkAndPunctTokenizer()
        self.sess.graph.finalize()

    def _preprocess(self, paragraphs: List[WebParagraph]) -> List[WebParagraph]:
        if self.model.preprocessor is not None:
            prepped = []
            for para in paragraphs:
                if hasattr(para, "spans"):
                    spans = para.spans
                else:
                    spans = None

                text, _, inv = self.model.preprocessor.encode_paragraph([], para.text, para.start == 0,
                                                                          np.zeros((0, 2), dtype=np.int32), spans)
                prepped.append(WebParagraph(
                    [text], para.original_text, inv, para.paragraph_num,
                    para.start, para.end,
                    para.source_name, para.source_url
                ))
            return prepped
        else:
            return paragraphs

    async def answer_question_spans(self, question: str) -> Tuple[np.ndarray, np.ndarray, List[WebParagraph]]:
        """
        Answer a question using web search, return the top spans and confidence scores for each paragraph
        """

        paragraphs = await self.get_question_context(question)
        question = self.tokenizer.tokenize_paragraph_flat(question)
        paragraphs = self._preprocess(paragraphs)
        t0 = time.perf_counter()
        qa_pairs = [ParagraphAndQuestion(c.get_context(), question, None, "") for c in paragraphs]
        encoded = self.model.encode(qa_pairs, False)
        spans, scores = self.sess.run([self.span, self.score], encoded)
        self.log.info("Computing answer spans took %.5f seconds" % (time.perf_counter() - t0))
        return spans, scores, paragraphs

    async def answer_question(self, question: str) -> Tuple[np.ndarray, List[WebParagraph]]:
        """
        Answer a question using web search, return the paragraphs and per-span confidence scores
        in the form of a (batch, max_num_context_tokens, max_num_context_tokens) array
        """

        self.log.info("Answering question \"%s\" with web search" % question)
        context = await self.get_question_context(question)
        question = self.tokenizer.tokenize_paragraph_flat(question)
        t0 = time.perf_counter()
        out = self._get_span_scores(question, context)
        self.log.info("Computing answer spans took %.5f seconds" % (time.perf_counter() - t0))
        return out

    def answer_with_doc(self, question: str, doc: str) -> Tuple[np.ndarray, List[WebParagraph]]:
        """ Answer a question using the given text as a document """

        self.log.info("Answering question \"%s\" with a given document" % question)
        # Tokenize
        question = self.tokenizer.tokenize_paragraph_flat(question)
        context = [self.tokenizer.tokenize_with_inverse(x, False) for x in self._split_regex.split(doc)]

        # Split into super-paragraphs
        context = self._split_document(context, "User", None)

        # Select top paragraphs
        context = self.paragraph_selector.prune(question, context)
        if len(context) == 0:
            raise ValueError("Unable to process documents")

        # Select the top answer span
        t0 = time.perf_counter()
        span_scores = self._get_span_scores(question, context)
        self.log.info("Computing answer spans took %.5f seconds" % (time.perf_counter() - t0))
        return span_scores

    def _get_span_scores(self, question: List[str], paragraphs: List[WebParagraph]):
        paragraphs = self._preprocess(paragraphs)
        qa_pairs = [ParagraphAndQuestion(c.get_context(), question, None, "") for c in paragraphs]
        encoded = self.model.encode(qa_pairs, False)
        return self.sess.run(self.span_scores, encoded), paragraphs

    def _split_document(self, para: List[ParagraphWithInverse], source_name: str,
                        source_url: Optional[str]):
        tokenized_paragraphs = []
        on_token = 0
        for i, para in enumerate(self.paragraph_splitter.split_inverse(para)):
            n_tokens = para.n_tokens
            tokenized_paragraphs.append(WebParagraph(
                para.text, para.original_text, para.spans, i + 1,
                on_token, on_token + n_tokens,
                source_name, source_url
            ))
            on_token += n_tokens
        return tokenized_paragraphs

    async def _tagme(self, question: str):
        payload = {"text": question,
                   "long_text": 3,
                   "lang": "en",
                   "gcube-token": self.tagme_api_key}
        async with self.client_sess.get(url=TAGME_API, params=payload) as resp:
            data = await resp.json()
        return [ann_json for ann_json in data["annotations"] if "title" in ann_json]

    async def get_question_context(self, question: str) -> List[WebParagraph]:
        """
        Find a set of paragraphs from the web that are relevant to the given question
        """

        tokenized_paragraphs = []
        if self.tagme_threshold is not None:
            self.log.info("Query tagme for %s", question)
            tags = await self._tagme(question)
            t0 = time.perf_counter()
            found = set()
            for tag in tags:
                if tag["rho"] >= self.tagme_threshold:
                    title = tag["title"]
                    if title in found:
                        continue
                    found.add(title)
                    doc = await self.wiki_corpus.get_wiki_article(title)
                    tokenized_paragraphs += self._split_document(doc.paragraphs,
                                                                 "Wikipedia: " + doc.title, doc.url)
            if len(tokenized_paragraphs) > 0:
                self.log.info("Getting wiki docs took %.5f seconds" % (time.perf_counter() - t0))

        if self.n_web_docs > 0:
            t0 = time.perf_counter()
            self.log.info("Running bing search for %s", question)
            search_results = await self.searcher.run_search(question, self.n_web_docs)
            t1 = time.perf_counter()
            self.log.info("Completed bing search, took %.5f seconds" % (t1 - t0))
            t0 = t1
            url_to_result = {x["url"]: x for x in search_results}
            self.log.info("Extracting text for %d results", len(search_results))
            text_docs = await self.text_extractor.get_text([x["url"] for x in search_results])

            for doc in text_docs:
                if len(doc.text) == 0:
                    continue
                search_r = url_to_result[doc.url]
                if self.blacklist_trivia_sites:
                    lower = search_r["displayUrl"].lower()
                    if 'quiz' in lower or 'trivia' in lower or 'answer' in lower:
                        # heuristic to ignore trivia sites, recommend by Mandar
                        self.log.debug("Skipping trivia site: " + lower)
                        continue

                paras_text = self._split_regex.split(doc.text.strip())

                paras_tokenized = [self.tokenizer.tokenize_with_inverse(x) for x in paras_text]

                tokenized_paragraphs += self._split_document(paras_tokenized, search_r["displayUrl"], doc.url)

            self.log.info("Completed extracting text, took %.5f seconds." % (time.perf_counter() - t0))

        self.log.info("Have %d paragraphs", len(tokenized_paragraphs))
        if len(tokenized_paragraphs) == 0:
            return []
        question = self.tokenizer.tokenize_sentence(question)
        return self.paragraph_selector.prune(question, tokenized_paragraphs)

    def close(self):
        if self.wiki_corpus is not None:
            self.wiki_corpus.close()
        if self.searcher is not None:
            self.searcher.close()
        self.sess.close()
        self.client_sess.close()
