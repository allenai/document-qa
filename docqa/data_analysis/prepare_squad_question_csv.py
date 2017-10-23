import json
from collections import namedtuple
from os.path import join

import numpy as np

from docqa.config import SQUAD_SOURCE_DIR

ParagraphAndQuestion = namedtuple('ParagraphAndQuestion', ['article_title', 'paragraph', 'question', 'answers', 'question_id'])


def init_google_csv(seed, n_samples, output_file, source):
    with open(source, 'r') as f:
        source_data = json.load(f)
    questions = []
    for article_ix, article in enumerate(source_data['data']):
        title = article["title"]
        for para_ix, para in enumerate(article['paragraphs']):
            text = para["context"]
            for question_ix, question in enumerate(para['qas']):
                q_id = question['id']
                questions.append(ParagraphAndQuestion(title, text, question["question"],
                                                      question['answers'], q_id))

    questions = sorted(questions, key=lambda x: x.question_id)
    np.random.RandomState(seed).shuffle(questions)

    with open(output_file, 'w') as f:
        f.write("question_id\tquestion\tanswer\tarticle_title\tcontext\n")
        for q in questions[:n_samples]:

            marked = q.paragraph
            for ans in q.answers[::-1]:
                start = ans["answer_start"]
                end = start + len(ans["text"])
                marked = marked[:end] + "}}}" + marked[end:]
                marked = marked[:start] + "{{{" + marked[start:]

            f.write(q.question_id)
            f.write("\t")
            f.write("\"" + q.question + "\"")
            f.write("\t")
            f.write("\"" + q.answers[0]["text"] + "\"")
            f.write("\t")
            f.write("\"" + q.article_title + "\"")
            f.write("\t")
            f.write(marked)
            f.write("\n")

if __name__ == "__main__":
    init_google_csv(0, 500, "/tmp/annotations.tsv", join(SQUAD_SOURCE_DIR, "train-v1.1.json"))
