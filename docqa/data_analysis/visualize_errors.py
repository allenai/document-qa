import argparse
import json

import numpy as np
from data_processing.span_utils import compute_span_f1

from squad.build_squad_data import SquadCorpus
from squad.squad_official_evaluation import f1_score as text_f1_score
from utils import flatten_iterable


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

    answers = []
    with open(args.answers, "r") as f:
        for line in f:
            answers.append(json.loads(line))

    dev_data = {x.question_id:x for x in data.get_dev()}

    np.random.shuffle(answers)
    correct_sent = 0
    total_f1 = 0

    for prediction in answers:
        point = dev_data[prediction["question_id"]]
        prediction = prediction["span_predictions"]
        start, end = prediction["start"], prediction["end"]

        context = flatten_iterable(point.context)
        text = origin_mapping.get_raw_text(point.article_id, point.paragraph_num, start, end)

        span_f1 = 0
        text_f1 = 0
        for ans in point.answer:
            span_f1 = max(span_f1, compute_span_f1((ans.para_word_start, ans.para_word_end), (start, end)))
            text_f1 = max(text_f1, text_f1_score(text, ans.text))

        total_f1 += text_f1

        ans_sent = 0
        offset = 0
        while end >= offset+len(point.context[ans_sent]):
            offset += len(point.context[ans_sent])
            ans_sent += 1

        incorrect_sent = text_f1 < 1 and all(ans_sent != ans.sent_end for ans in point.answer)
        correct_sent += not incorrect_sent

        if incorrect_sent:
            context = list(context)
            for ans in point.answer:
                if not context[ans.para_word_start].startswith(bcolors.CORRECT):
                    context[ans.para_word_start] = bcolors.CORRECT + context[ans.para_word_start]
                    context[ans.para_word_end] = context[ans.para_word_end] + bcolors.ENDC
            print()
            print(" ".join(point.question))
            distractor = list(point.context[ans_sent])
            distractor[start-offset] = bcolors.ERROR + distractor[start-offset]
            distractor[end - offset] = distractor[end - offset] + bcolors.ENDC
            print(" ".join(distractor))
            # print(text)
            # context[start] = bcolors.ERROR + context[start]
            # context[end] = context[end] + bcolors.ENDC
            # print(" ".join(context))
            # input()
    print(total_f1 / len(answers))
    print(correct_sent / len(answers))


if __name__ == "__main__":
    main()