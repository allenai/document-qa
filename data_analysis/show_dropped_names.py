import tensorflow as tf
import numpy as np

from data_processing.paragraph_qa import split_docs
from data_processing.qa_data import QaCorpusLazyStats, compute_voc, ParagraphAndQuestionSpec
from encoder import DocumentAndQuestionEncoder, SingleSpanAnswerEncoder
from nn.embedder import DropNames
from squad.build_squad_dataset import SquadCorpus


def main():
    embed = DropNames(vec_name="glove.840B.300d", word_vec_init_scale=0, learn_unk=False,
              keep_probs=0, kind="shuffle")
    corpus = SquadCorpus()
    print("Loading...")
    docs = corpus.get_train()
    data = split_docs(docs)
    print("Get voc...")
    voc = compute_voc(data)
    print("Init ...")
    stats = QaCorpusLazyStats(data)
    loader = corpus.get_resource_loader()
    embed.set_vocab(stats, loader, [])
    embed.init(loader, voc)

    print("Init encoder")

    ix_to_word = {ix:w for w, ix in embed._word_to_ix.items()}
    ix_to_word[1] = "UNK"
    ix_to_word[0] = "PAD"

    encoder = DocumentAndQuestionEncoder(SingleSpanAnswerEncoder())
    encoder.init(ParagraphAndQuestionSpec(1, None, None, None, None, None), False, embed, None)

    sess = tf.Session()

    np.random.shuffle(data)
    for q in data:
        encoded = encoder.encode([q], True)
        context_words, question_words = [encoded[encoder.context_words], encoded[encoder.question_words]]
        print([ix_to_word[i] for i in question_words[0]])
        context_words, question_words = embed.drop_names([context_words, question_words])
        print([ix_to_word[i] for i in sess.run(question_words)[0]])


# Months,
# Days,
# Locations
#




if __name__ == "__main__":
    main()