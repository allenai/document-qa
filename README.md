## Document QA
This repo contains code to train neural question answering models in tensorflow, 
and in particular for the cases when we want to run the model over multiple paragraphs for 
each questions. Code is included to train on the TriviaQA and SQuAD datasets.

## Setup
### Dependencies
We require python >= 3.5, tensorflow 1.3, and a handful of other supporting libraries. 
To install the dependencies (beside tensorflow) use

`pip install -r requirements.txt`

The stopword corpus and punkt sentence tokenizer for nltk will are needed and can be fetched with:
 
 `python -m nltk.downloader punkt stopwords`
 
### Source Data
By default, we expect source data to stored in "~/data" and preprocessed data will be 
dump to "./data". The expected file locations can be changed by altering config.py.
 
#### Word Vectors
The models we train use the common crawl 840 billion token GloVe word vectors. 
They are expected to exist in "~/data/glove/glove.840B.300d.txt" or "~/data/glove/glove.840B.300d.txt.gz"

#### SQuAD Data
Training or testing on SQuAD requires downloading the SQuAD train/test files. We 
expect the data to be tokenized as a pre-preprocessing step, which can be done 
by running 

`python docqa/squad/build_squad_dataset.py -s /path/to/squad/directory``


#### TriviaQA Data
The raw TriviaQA data is expected to be unzipped in "~/data/triviaqa". Training
or testing in the unfiltered setting requires the unfiltered data to be 
download to "~/data/triviaqa-unfiltered".

To use TriviaQA we need to tokenize the evidence documents, which can be done by

`python docqa/triviaqa/evidence_corpus.py`

This can be slow, we support multi-processing

`python docqa/triviaqa/evidence_corpus.py --n_processes 8`

Finally we need to tokenize the questions and locate the relevant 
answers spans in each document. Run

`python docqa/triviaqa/build_span_corpus.py -n {web|wiki|open}`

to build the desired document set.
 

## Training

## Testing
 