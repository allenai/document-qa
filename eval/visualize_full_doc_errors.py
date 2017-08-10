import argparse
import json

import numpy as np
from nltk.corpus import stopwords

from eval.full_document_eval import QuestionAnswer
from squad.squad import SquadCorpus
from squad.squad_official_evaluation import f1_score as text_f1_score


class bcolors:
    CORRECT = '\033[94m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("answers")
    args = parser.parse_args()

    data = SquadCorpus()
    origin_mapping = data.get_original_text_mapping()

    stop = set(stopwords.words('english'))

    with open(args.answers, "r") as f:
        answers = [QuestionAnswer(**x) for x in json.load(f)]

    dev_data = {x.question_id:x for x in data.get_dev()}
    paragraph_map = {}
    for p in dev_data.values():
        paragraph_map[(p.article_id, p.paragraph_num)] = p.context

    np.random.shuffle(answers)
    # tmp = open("/tmp/tmp.csv", "w")

    for prediction in answers:
        point = dev_data[prediction.question_id]
        start, end = prediction.doc_span

        context = paragraph_map[(point.article_id, prediction.paragraph_num)]
        text = origin_mapping.get_raw_text(point.article_id, prediction.paragraph_num, start, end)

        text_f1 = 0
        for ans in point.answer:
            text_f1 = max(text_f1, text_f1_score(text, ans.text))

        ans_sent = 0
        offset = 0
        while end >= offset+len(context[ans_sent]):
            offset += len(context[ans_sent])
            ans_sent += 1
        sent_start = start-offset
        sent_end = end - offset

        question_words = set(x.lower() for x in point.question if x.lower() not in stop)

        if prediction.paragraph_num != point.paragraph_num and text_f1 == 0:
            # tmp.write(" ".join(point.question))
            # tmp.write("\t" + point.article_title)
            # tmp.write("\t" + text)
            # tmp.write("\t" + str(list(set(x.text for x in point.answer))))
            # # tmp.write("\t" + " ".join(context[ans_sent]))
            #
            distractor = list(context[ans_sent])
            # distractor[sent_start] = "{{{" + distractor[sent_start]
            # distractor[sent_end] = distractor[sent_end] + "}}}"
            #
            # tmp.write("\t" + " ".join(distractor))
            # tmp.write("\n")


            # print(" ".join(point.question))
            # context = list(context)
            # for ans in point.answer:
            #     if not context[ans.para_word_start].startswith(bcolors.CORRECT):
            #         context[ans.para_word_start] = bcolors.CORRECT + context[ans.para_word_start]
            #         context[ans.para_word_end] = context[ans.para_word_end] + bcolors.ENDC
            print(" ".join(point.question) + " " + str(list(set(x.text for x in point.answer))))
            print("Article=%s, Prediction=%s, Correct=%s" % (point.article_title, text, str(list(set(x.text for x in point.answer)))))

            for i, word in enumerate(distractor):
                if (i < sent_start or i > sent_end) and word.lower() in question_words:
                    distractor[i] = bcolors.CORRECT + word + bcolors.ENDC

            print(" ".join(distractor))
            print(" ".join(point.context[point.answer[0].sent_start]))
            input()
    # print(total_f1 / len(answers))
    # print(correct_sent / len(answers))


if __name__ == "__main__":
    main()