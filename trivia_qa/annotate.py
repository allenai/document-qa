import numpy as np


"""
Mess of various annotations considered
"""


class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    GREEN = '\033[92m'
    ENDC = '\033[0m'


def annotate_first_tokens(questions, file_map, output):
    have = set()
    if exists(output):
        with open(output, "r") as f:
            for line in f:
                parts = line.split("\t")
                have.add((parts[0], parts[1]))  # (quid, doc)
    else:
        with open(output, "w") as f:
            f.write("\t".join(["quid", "doc_id", "answerable"]))
            f.write("\n")

    sent_tokenize = nltk.sent_tokenize
    word_tokenize = nltk.word_tokenize
    np.random.shuffle(questions)
    stop = set(stopwords.words('english'))
    stop.update(["many", "how", "?", ",", "-", "."])

    with open(output, "a") as o:
        for question in questions:
            question_tokens = set(x.lower() for x in word_tokenize(question.question) if x.lower() not in stop)
            regex = answer_regex(question.answer.normalized_aliases)
            for doc in question.web_docs:
                if not doc.trivia_qa_selected:
                    continue
                if (question.question_id, doc.doc_id) in have:
                    continue
                print(question.question + " " + str(question.answer.value))
                print(str(question.answer.normalized_aliases))
                print(doc.url + " " + doc.title)

                with open(TRIVIA_QA + "/evidence/web/" + file_map[doc.url], "r") as f:
                    full_text = f.read().strip()

                tokenized = flatten_iterable([word_tokenize(s) for s in sent_tokenize(full_text)])
                for i, token in enumerate(tokenized):
                    if token.lower() in question_tokens:
                        tokenized[i] = bcolors.GREEN + token + bcolors.ENDC

                text = " ".join(tokenized[:800])

                para, count = regex.subn(bcolors.CORRECT + "\\1" + bcolors.ENDC, text)
                full_text_count = len(regex.findall(full_text))

                print("%d Matches, (%d in full text)" % (count, full_text_count))
                print(para)
                while True:
                    label_str = input().strip()
                    # d -> directly answers (i.e. in a sentence)
                    # l -> "long" answer information their, but distributed in doc
                    # i -> "indirect" relevant information their, but only loosely or partially answer the question
                    # n -> "none" answer is a rare entity, not relveant iformation is given
                    # e -> "empty" answer simply does not appear in the text
                    if label_str[0] in ["d", "l", "i", "e", "n"] and (len(label_str) == 1 or label_str[1] in ["p", "c"]):
                        break
                    else:
                        print("Uknown: " + label_str)

                o.write("\t".join(str(x) for x in [question.question_id, doc.doc_id, label_str]))
                o.write("\n")


def show_annotations():
    annotations = {}
    with open("/tmp/annotations.tsv", "r") as f:
        for line in f:
            parts = line.split("\t")
            annotations[(parts[0], parts[1], parts[2])] = parts[3]
    print(len(annotations))
    print(len(set(x[0] for x in annotations)))
    print(len(set((x[0], x[1]) for x in annotations)))

