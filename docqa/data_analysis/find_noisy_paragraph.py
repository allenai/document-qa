from docqa.data_processing.document_splitter import TopTfIdf, MergeParagraphs
from docqa.data_processing.text_utils import NltkPlusStopWords
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset
from docqa.utils import flatten_iterable


def main():
    data = TriviaQaWebDataset()

    stop = NltkPlusStopWords()
    splitter = MergeParagraphs(400)
    selector = TopTfIdf(stop, 4)

    print("Loading data..")
    train = data.get_train()
    print("Start")
    for q in train:
        for doc in q.all_docs:
            if len(doc.answer_spans) > 3:
                text = splitter.split_annotated(data.evidence.get_document(doc.doc_id), doc.answer_spans)
                text = selector.prune(q.question, text)
                for para in text:
                    if len(para.answer_spans) > 3:
                        print(q.question)
                        text = flatten_iterable(para.text)
                        for s,e in para.answer_spans:
                            text[s] = "{{{" + text[s]
                            text[e] = text[e] + "}}}"
                        print(" ".join(text))
                        input()

if __name__ == "__main__":
    main()