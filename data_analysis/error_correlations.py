"""
Explore errors caused by our tokenziation and/or token cleaning
"""
import argparse
import json
import re
from collections import Counter

from squad.squad import SquadCorpus

from data_processing.span_data import compute_span_f1
from data_processing.text_features import is_number
from squad.squad_official_evaluation import f1_score as text_f1_score


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("answers")
    parser.add_argument("output")
    args = parser.parse_args()
    vecs = SquadCorpus().get_resource_loader().load_word_vec("glove.6B.100d")

    collection = SquadCorpus().get_train_corpus()[1]
    counts = collection.get_question_counts() + collection.get_document_counts()
    lower_counts = Counter()
    for w,c in counts.items():
        lower_counts[w.lower()] += c

    def is_name(word):
        if word[0].isupper() and word[1:].islower():
            lower = word.lower()
            if lower not in lower_counts:
                return True
            if counts[word] / lower_counts[lower] > 0.9:
                return True
        return False


    answers = []
    with open(args.answers, "r") as f:
        for line in f:
            answers.append(json.loads(line))

    dev_data = {x.question_id: x for x in SquadCorpus().get_dev()}

    stop = {",", "?", "\'", "the", "a", "is", "of"}

    top_question_words = Counter()
    for point in dev_data.values():
        top_question_words.update([x.lower() for x in point.question if x.lower() not in stop])
    top_question_words = set(x[0] for x in top_question_words.most_common(25))

    alpha = re.compile("^[a-z]+$")

    all_features = []
    for prediction in answers:
        q_id = prediction["question_id"]
        point = dev_data[q_id]

        prediction = prediction["bound-17-span-predictions"]
        start, end = prediction["start"], prediction["end"]

        val = prediction["val"]
        text = point.get_original_text(start, end)

        span_f1 = 0
        text_f1 = 0
        for ans in point.answer:
            span_f1 = max(span_f1, compute_span_f1((ans.para_word_start, ans.para_word_end), (start, end)))
            text_f1 = max(text_f1, text_f1_score(text, ans.text))

        any_unk = 0
        any_num = 0
        question_name = 0
        question_alpha_unk = 0

        answer_num = 0
        answer_unk = 0
        answer_name = 0
        answer_alpha_unk = 0

        features = {"HasWord-"+x.lower(): 1 for x in point.question if x.lower() in top_question_words}

        for word in point.question:
            lower_word = word.lower()
            if lower_word not in vecs:
                any_unk += 1
                if alpha.match(lower_word):
                    question_alpha_unk += 1
            if is_number(lower_word):
                any_num += 1
            if is_name(word):
                question_name += 1

        for word in point.answer[0].text.split():
            lower_word = word.lower()
            if lower_word not in vecs:
                answer_unk += 1
                if alpha.match(lower_word):
                    answer_alpha_unk += 1
            if is_name(word):
                answer_name += 1

            if is_number(lower_word):
                answer_num += 1


        answer_set = set(x.text for x in point.answer)

        features.update(dict(
            question="\"" + " ".join(point.question).replace("\"", "``") + "\"",
            answer_text="\"" + str(list(set(x.text for x in point.answer))).replace("\"", "``") + "\"",
            predicted_text="\"" + text.replace("\"", "``") + "\"",
            span_f1=span_f1,
            text_f1=text_f1,
            predicted_prob=val,
            question_id=q_id,
            question_unk=any_unk,
            question_num=any_num,
            answer_name=answer_name,
            question_name=question_name,
            question_alpha_unk=question_alpha_unk,
            answer_num=answer_num,
            answer_unk=answer_unk,
            answer_alpha_unk=answer_alpha_unk,
            num_answers=len(answer_set)
        ))
        all_features.append(features)

    keys = set()
    for feature_dict in all_features:
        keys.update(feature_dict)
    keys = list(keys)

    with open(args.output, "w") as f:
        f.write("\t".join(keys))
        f.write("\n")
        for feature_dict in all_features:
            f.write("\t".join(str(feature_dict.get(k, 0)) for k in keys))
            f.write("\n")


if __name__ == "__main__":
    main()