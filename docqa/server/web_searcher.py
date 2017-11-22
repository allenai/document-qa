from typing import List, Dict, Optional

import logging

import ujson
import asyncio
from aiohttp import ClientSession
from os.path import exists

BING_API = "https://api.cognitive.microsoft.com/bing/"


class AsyncWebSearcher(object):
    """ Runs search requests and returns the results """

    def __init__(self, bing_api, bing_version, loop=None):
        if bing_api is None or not isinstance(bing_api, str):
            raise ValueError("Need a string Bing API key")
        self.bing_api = bing_api
        self.url = BING_API + bing_version + "/search"
        self.cl_sess = ClientSession(headers={"Ocp-Apim-Subscription-Key": self.bing_api}, loop=loop)

    async def run_search(self, question: str, n_docs: int) -> List[Dict]:
        # avoid quoting the entire question, some triviaqa questions have this form
        # TODO is this the right place to do this?
        question = question.strip("\"\' ")
        async with self.cl_sess.get(url=self.url, params=dict(count=n_docs, q=question, mkt="en-US")) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise ValueError("Web search error %s" % data)

        if "webPages" not in data:
            return []
        else:
            return data["webPages"]["value"]

    def close(self):
        self.cl_sess.close()


class ExtractedWebDoc(object):
    def __init__(self, ur: str, text: str):
        self.url = ur
        self.text = text


class AsyncBoilerpipeCliExtractor(object):
    """
    Downloads documents from URLs and returns the extracted text

    TriviaQA used boilerpipe (https://github.com/kohlschutter/boilerpipe) to extract the
    "main" pieces of text from web documents. There is, far as I can tell, no complete
    python re-implementation so far the moment we shell out to a jar file (boilerpipe.jar)
    which downloads files from the given URLs and runs them through boilerpipe's extraction code
    using multiple threads.
    """

    JAR = "docqa/server/boilerpipe.jar"

    def __init__(self, n_threads: int=10, timeout: int=None,
                 process_additional_timeout: Optional[int]=5):
        """
        :param n_threads: Number of threads to use when downloading urls
        :param timeout: Time to wait while downloading urls, if the time limit is reached
                        downloads that are still hanging will be returned as errors
        :param process_additional_timeout: How long to wait for the downloading sub-process to return,
                                           in addition to `timeout`. If this timeout is hit no results will
                                           be returned, so this is a last-resort to stop the server from freezing
        """
        self.log = logging.getLogger('downloader')
        if not exists(self.JAR):
            raise ValueError("Could not find boilerpipe jar")
        self.timeout = timeout
        self.n_threads = n_threads
        if self.timeout is None:
            self.proc_timeout = None
        else:
            self.proc_timeout = timeout + process_additional_timeout

    async def get_text(self, urls: List[str]) -> List[ExtractedWebDoc]:
        process = await asyncio.create_subprocess_exec(
            "java", "-jar", self.JAR, *urls, "-t", str(self.n_threads),
            "-l", str(self.timeout),
            stdout=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(),
                                                timeout=self.proc_timeout)
        text = stdout.decode("utf-8")
        data = ujson.loads(text)
        ex = data["extracted"]
        errors = data["error"]
        if len(errors) > 0:
            self.log.info("%d extraction errors: %s" % (len(errors), str(list(errors.items()))))
        return [ExtractedWebDoc(url, ex[url]) for url in urls if url in ex]
