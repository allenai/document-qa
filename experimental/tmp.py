from data_processing.text_utils import get_paragraph_tokenizer


def main():
    sent_tok, word_tok = get_paragraph_tokenizer("NLTK_AND_CLEAN")
    print(" ".join(word_tok("Hi There\"")))

if __name__ == "__main__":
    main()