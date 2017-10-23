import numpy as np
from docqa.squad.squad_data import SquadCorpus, split_docs


def main():
    data = split_docs(SquadCorpus().get_train())
    np.random.shuffle(data)
    for point in data:
        print(" ".join(point.question))


if __name__ == "__main__":
    main()