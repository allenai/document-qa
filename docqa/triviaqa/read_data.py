import ujson as json
import unicodedata
from os.path import join
from typing import List

from docqa.triviaqa.trivia_qa_eval import normalize_answer as triviaqa_normalize_answer

"""
Read and represent trivia-qa data 
"""


def normalize_wiki_filename(filename):
    """
    Wiki filenames have been an pain, since the data seems to have filenames encoded in
    the incorrect case sometimes, and we have to be careful to keep a consistent unicode format.
    Our current solution is require all filenames to be normalized like this
    """
    return unicodedata.normalize("NFD", filename).lower()


class WikipediaEntity(object):
    __slots__ = ["value", "normalized_value", "aliases", "normalized_aliases",
                 "wiki_entity_name", "normalized_wiki_entity_name", "human_answers"]

    def __init__(self, value: str, normalized_value: str, aliases, normalized_aliases: List[str],
                 wiki_entity_name: str, normalized_wiki_entity_name: str, human_answers):
        self.aliases = aliases
        self.value = value
        self.normalized_value = normalized_value
        self.normalized_aliases = normalized_aliases
        self.wiki_entity_name = wiki_entity_name
        self.normalized_wiki_entity_name = normalized_wiki_entity_name
        self.human_answers = human_answers

    @property
    def all_answers(self):
        if self.human_answers is None:
            return self.normalized_aliases
        else:
            # normalize to be consistent with the other normallized aliases
            human_answers = [triviaqa_normalize_answer(x) for x in self.human_answers]
            return self.normalized_aliases + [x for x in human_answers if len(x) > 0]

    def __repr__(self) -> str:
        return self.value


class Numerical(object):
    __slots__ = ["number", "aliases", "normalized_aliases", "value", "unit",
                 "normalized_value", "multiplier", "human_answers"]

    def __init__(self, number: float, aliases, normalized_aliases, value, unit,
                 normalized_value, multiplier, human_answers):
        self.number = number
        self.aliases = aliases
        self.normalized_aliases = normalized_aliases
        self.value = value
        self.unit = unit
        self.normalized_value = normalized_value
        self.multiplier = multiplier
        self.human_answers = human_answers

    @property
    def all_answers(self):
        if self.human_answers is None:
            return self.normalized_aliases
        else:
            human_answers = [triviaqa_normalize_answer(x) for x in self.human_answers]
            return self.normalized_aliases + [x for x in human_answers if len(x) > 0]

    def __repr__(self) -> str:
        return self.value


class FreeForm(object):
    __slots__ = ["value", "normalized_value", "aliases", "normalized_aliases", "human_answers"]

    def __init__(self, value, normalized_value, aliases, normalized_aliases, human_answers):
        self.value = value
        self.aliases = aliases
        self.normalized_value = normalized_value
        self.normalized_aliases = normalized_aliases
        self.human_answers = human_answers

    @property
    def all_answers(self):
        if self.human_answers is None:
            return self.normalized_aliases
        else:
            human_answers = [triviaqa_normalize_answer(x) for x in self.human_answers]
            return self.normalized_aliases + [x for x in human_answers if len(x) > 0]

    def __repr__(self) -> str:
        return self.value


class Range(object):
    __slots__ = ["value", "normalized_value", "aliases", "normalized_aliases",
                 "start", "end", "unit", "multiplier", "human_answers"]

    def __init__(self, value, normalized_value, aliases, normalized_aliases,
                 start, end, unit, multiplier, human_answers):
        self.value = value
        self.normalized_value = normalized_value
        self.aliases = aliases
        self.normalized_aliases = normalized_aliases
        self.start = start
        self.end = end
        self.unit = unit
        self.multiplier = multiplier
        self.human_answers = human_answers

    @property
    def all_answers(self):
        if self.human_answers is None:
            return self.normalized_aliases
        else:
            human_answers = [triviaqa_normalize_answer(x) for x in self.human_answers]
            return self.normalized_aliases + [x for x in human_answers if len(x) > 0]

    def __repr__(self) -> str:
        return self.value


class TagMeEntityDoc(object):
    __slots__ = ["rho", "link_probability", "title", "trivia_qa_selected", "answer_spans"]

    def __init__(self, rho, link_probability, title):
        self.rho = rho
        self.link_probability = link_probability
        self.title = title
        self.trivia_qa_selected = False
        self.answer_spans = None

    @property
    def doc_id(self):
        return self.title

    def __repr__(self) -> str:
        return "TagMeEntityDoc(%s)" % self.title


class SearchEntityDoc(object):
    __slots__ = ["title", "trivia_qa_selected", "answer_spans"]

    def __init__(self, title):
        self.title = title
        self.answer_spans = None
        self.trivia_qa_selected = False

    @property
    def doc_id(self):
        return self.title

    def __repr__(self) -> str:
        return "SearchEntityDoc(%s)" % self.title


class SearchDoc(object):
    __slots__ = ["title", "description", "rank", "url", "trivia_qa_selected", "answer_spans"]

    def __init__(self, title, description, rank, url):
        self.title = title
        self.description = description
        self.rank = rank
        self.url = url
        self.answer_spans = None
        self.trivia_qa_selected = False

    @property
    def doc_id(self):
        return self.url

    def __repr__(self) -> str:
        return "SearchDoc(%s)" % self.title


class TriviaQaQuestion(object):
    __slots__ = ["question", "question_id", "answer", "entity_docs", "web_docs"]

    def __init__(self, question, question_id, answer, entity_docs, web_docs):
        self.question = question
        self.question_id = question_id
        self.answer = answer
        self.entity_docs = entity_docs
        self.web_docs = web_docs

    @property
    def all_docs(self):
        if self.web_docs is not None:
            return self.web_docs + self.entity_docs
        else:
            return self.entity_docs

    def to_compressed_json(self):
        return [
            self.question,
            self.question_id,
            [self.answer.__class__.__name__] + [getattr(self.answer, x) for x in self.answer.__slots__],
            [[doc.__class__.__name__] + [getattr(doc, x) for x in doc.__slots__] for doc in self.entity_docs],
            [[getattr(doc, x) for x in doc.__slots__] for doc in self.web_docs],
        ]

    @staticmethod
    def from_compressed_json(text):
        question, quid, answer, entity_docs, web_docs = json.loads(text)
        if answer[0] == "WikipediaEntity":
            answer = WikipediaEntity(*answer[1:])
        elif answer[0] == "Numerical":
            answer = Numerical(*answer[1:])
        elif answer[0] == "FreeForm":
            answer = FreeForm(*answer[1:])
        elif answer[0] == "Range":
            answer = Range(*answer[1:])
        else:
            raise ValueError()
        for i, doc in enumerate(entity_docs):
            if doc[0] == "TagMeEntityDoc":
                entity_docs[i] = TagMeEntityDoc(*doc[1:])
            elif doc[0] == "SearchEntityDoc":
                    entity_docs[i] = SearchEntityDoc(*doc[1:])
        web_docs = [SearchDoc(*x) for x in web_docs]
        return TriviaQaQuestion(question, quid, answer, entity_docs, web_docs)


def iter_question_json(filename):
    """ Iterates over trivia-qa questions in a JSON file, useful if the file is too large to be
    parse all at once """
    with open(filename, "r") as f:
        if f.readline().strip() != "{":
            raise ValueError()
        if "Data\": [" not in f.readline():
            raise ValueError()
        line = f.readline()
        while line.strip() == "{":
            obj = []
            line = f.readline()
            while not line.startswith("        }"):
                obj.append(line)
                line = f.readline()
            yield "{" + "".join(obj) + "}"
            if not line.startswith("        },"):
                # no comma means this was the last element of the data list
                return
            else:
                line = f.readline()
        else:
            raise ValueError()


def build_questions(json_questions, title_to_file, require_filename):
    for q in json_questions:
        q = json.loads(q)
        ans = q.get("Answer")
        valid_attempt = q.get("QuestionVerifiedEvalAttempt", False)
        if valid_attempt and not q["QuestionPartOfVerifiedEval"]:
            continue  # don't both with questions in the verified set that were rejected
        if ans is not None:
            answer_type = ans["Type"]
            if answer_type == "WikipediaEntity":
                answer = WikipediaEntity(ans["NormalizedValue"], ans["Value"], ans["Aliases"], ans["NormalizedAliases"],
                                         ans["MatchedWikiEntityName"], ans["NormalizedMatchedWikiEntityName"],
                                         ans.get("HumanAnswers"))
                if not (len(ans) == 7 or (len(ans) == 8 and "HumanAnswers" in ans)):
                    raise ValueError()
            elif answer_type == "Numerical":
                answer = Numerical(float(ans["Number"]), ans["Aliases"], ans["NormalizedAliases"],
                                   ans["Value"], ans["Unit"], ans["NormalizedValue"],
                                   ans["Multiplier"], ans.get("HumanAnswers"))
                if not (len(ans) == 8 or (len(ans) == 9 and "HumanAnswers" in ans)):
                    raise ValueError()
            elif answer_type == "FreeForm":
                answer = FreeForm(ans["Value"], ans["NormalizedValue"], ans["Aliases"],
                                  ans["NormalizedAliases"], ans.get("HumanAnswers"))
                if not (len(ans) == 5 or (len(ans) == 6 and "HumanAnswers" in ans)):
                    raise ValueError()
            elif answer_type == "Range":
                answer = Range(ans["Value"], ans["NormalizedValue"], ans["Aliases"], ans["NormalizedAliases"],
                               float(ans["To"]), float(ans["From"]), ans["Unit"],
                               ans["Multiplier"], ans.get("HumanAnswers"))
                if not (len(ans) == 9 or (len(ans) == 10 and "HumanAnswers" in ans)):
                    if "Number" in ans:
                        # This appears to be a bug, the number fields in this
                        # cases seem to be meaningless (and VERY rare)
                        pass
                    else:
                        raise ValueError()
            else:
                raise ValueError()
        else:
            answer = None

        entity_pages = []
        for page in q["EntityPages"]:
            verified_attempt = page.get("DocVerifiedEvalAttempt", False)
            if verified_attempt and not page["DocPartOfVerifiedEval"]:
                continue
            title = page["Title"]
            if page["DocSource"] == "Search":
                entity_pages.append(SearchEntityDoc(title))
            elif page["DocSource"] == "TagMe":
                entity_pages.append(TagMeEntityDoc(page.get("Rho"), page.get("LinkProbability"), title))
            else:
                raise ValueError()
            filename = page.get("Filename")
            if filename is not None:
                filename = join("wikipedia", filename[:filename.rfind(".")])
                filename = normalize_wiki_filename(filename)
                cur = title_to_file.get(title)
                if cur is None:
                    title_to_file[title] = filename
                elif cur != filename:
                    raise ValueError()
            elif require_filename:
                raise ValueError()

        if "SearchResults" in q:
            web_pages = []
            for page in q["SearchResults"]:
                verified_attempt = page.get("DocVerifiedEvalAttempt", False)
                if verified_attempt and not page["DocPartOfVerifiedEval"]:
                    continue
                url = page["Url"]
                web_pages.append(SearchDoc(page["Title"], page["Description"], page["Rank"], url))
                filename = page.get("Filename")
                if filename is not None:
                    filename = join("web", filename[:filename.rfind(".")])
                    cur = title_to_file.get(url)
                    if cur is None:
                        title_to_file[url] = filename
                    elif cur != filename:
                        raise ValueError()
                elif require_filename:
                    raise ValueError()
        else:
            web_pages = None

        yield TriviaQaQuestion(q["Question"], q["QuestionId"], answer, entity_pages, web_pages)


def iter_trivia_question(filename, file_map, require_filename):
    return build_questions(iter_question_json(filename), file_map, require_filename)


