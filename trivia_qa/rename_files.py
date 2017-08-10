from os import walk, listdir, rename

from os.path import join, isfile, isdir

import unicodedata

from trivia_qa.read_data import normalize_wiki_filename


def normalize_names(directory, dry_run=True):
    for filename in listdir(directory):
        filename = join(directory, filename)
        normalized_name = normalize_wiki_filename(filename)
        if filename != normalized_name:
            rename(filename, normalized_name)
        if isdir(normalized_name):
            normalize_names(normalized_name, dry_run)


def main():
    normalize_names("data/triviaqa/evidence/wikipedia")


if __name__ == "__main__":
    main()