from tqdm import tqdm

from docqa.squad.squad_data import SquadCorpus
from docqa.squad.squad_official_evaluation import f1_score
from docqa.utils import flatten_iterable

"""
Explore errors caused by our tokenziation and/or token cleaning
"""


def main():
    data = SquadCorpus()

    string_f1 = 0
    mapped_string_f1 = 0

    docs = data.get_train()
    n_questions = 0

    for doc in tqdm(docs):
        for para in doc.paragraphs:
            words = flatten_iterable(para.text)
            for question in para.questions:
                n_questions += 1
                span_answer = question.answer[0]
                span_str = " ".join(words[span_answer.para_word_start:span_answer.para_word_end+1])
                raw_answer = span_answer.text
                mapped_str = para.get_original_text(span_answer.para_word_start, span_answer.para_word_end)

                string_f1 += f1_score(raw_answer, span_str)
                mapped_string_f1 += f1_score(raw_answer, mapped_str)

    print(string_f1 / n_questions)
    print(mapped_string_f1 / n_questions)


if __name__ == "__main__":
    main()