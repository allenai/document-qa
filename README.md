## Document QA
This repo contains code for our paper [Simple and Effective Multi-Paragraph Reading Comprehension](https://arxiv.org/abs/1710.10723).
It can be used to  train neural question answering models in tensorflow, 
and in particular for the case when we want to run the model over multiple paragraphs for 
each question. Code is included to train on the [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) 
and [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) datasets.

A demo of this work can be found at [documentqa.allenai.org](https://documentqa.allenai.org)

Small forewarning, this is still much more of a research codebase then a library.
we anticipate porting this work in [allennlp](https://github.com/allenai/allennlp) where it will 
enjoy a cleaner implementation and more stable support.


## Setup
### Dependencies
We require python >= 3.5, tensorflow 1.3, and a handful of other supporting libraries. 
Tensorflow should be installed separately following the docs. To install the other dependencies use

`pip install -r requirements.txt`

The stopword corpus and punkt sentence tokenizer for nltk are needed and can be fetched with:
 
 `python -m nltk.downloader punkt stopwords`
 
The easiest way to run this code is to use:

``export PYTHONPATH=${PYTHONPATH}:`pwd` ``

### Data
By default, we expect source data to stored in "\~/data" and preprocessed data to be
stored int "./data". The expected file locations can be changed by altering config.py.
 

#### Word Vectors
The models we train use the common crawl 840 billion token GloVe word vectors from [here](https://nlp.stanford.edu/projects/glove/).
They are expected to exist in "\~/data/glove/glove.840B.300d.txt" or "\~/data/glove/glove.840B.300d.txt.gz".

For example:

```
mkdir -p ~/data
mkdir -p ~/data/glove
cd ~/data/glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
```

#### SQuAD Data
Training or testing on SQuAD requires downloading the SQuAD train/dev files into ~/data/squad.
This can be done as follows:

```
mkdir -p ~/data/squad
cd ~/data/squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

then running:

``python docqa/squad/build_squad_dataset.py``

This builds pkl files of the tokenized data in "./data/squad"

#### TriviaQA Data
The raw TriviaQA data is expected to be unzipped in "\~/data/triviaqa". Training
or testing in the unfiltered setting requires the unfiltered data to be 
download to "\~/data/triviaqa-unfiltered".

```
mkdir -p ~/data/triviaqa
cd ~/data/triviaqa
wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz
tar xf triviaqa-rc.tar.gz
rm triviaqa-rc.tar.gz

cd ~/data
wget http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
tar xf triviaqa-unfiltered.tar.gz
rm triviaqa-unfiltered.tar.gz
```

To use TriviaQA we need to tokenize the evidence documents, which can be done by

`python docqa/triviaqa/evidence_corpus.py`

This can be slow, we support multi-processing

`python docqa/triviaqa/evidence_corpus.py --n_processes 8`

This builds evidence files in "./data/triviaqa/evidence" that are split into 
paragraphs, sentences, and tokens. Then we need to tokenize the questions and locate the relevant 
answers spans in each document. Run

`python docqa/triviaqa/build_span_corpus.py {web|wiki|open} --n_processes 8`

to build the desired set. This builds pkl files "./data/triviaqa/{web|wiki|open}"


## Training
Once the data is in place our models can be trained by

`python docqa/scripts/ablate_{triviaqa|squad|triviaqa_unfiltered}.py`


See the help menu for these scripts for more details. Note that since we use the Cudnn RNN implementations,
these models can only be trained on a GPU. We do provide a script for converting
the (trained) models to CPU versions:

`python docqa/scripts/convert_to_cpu.py`

Modifying the hyper-parameters beyond the ablations requires building your own train script.

## Testing
### SQuAD
Use "docqa/eval/squad_eval.py" to evaluate on paragraph-level (i.e., standard) SQuAD. For example:

`python docqa/eval/squad_eval.py -o output.json -c dev /path/to/model/directory`

"output.json" can be used with the official evaluation script, for example:

`python docqa/squad/squad_official_evaluation.py ~/data/squad/dev-v1.1.json output.json`

Use "docqa/eval/squad_full_document_eval.py" to evaluate on the document-level. For example

`python docqa/eval/squad_full_document_eval.py -c dev /path/to/model/directory output.csv`

This will store the per-paragraph results in output.csv, we can then run:

`docqa/eval/ranked_squad_scores.py output.csv`

to get ranked scores as more paragraphs are used.


### TriviaQA
Use "docqa/eval/triviaqa_full_document_eval.py" to evaluate on TriviaQA datasets, like:
 
`python docqa/eval/triviaqa_full_document_eval.py --n_processes 8 -c web-dev --tokens 400 -f tfidf-15 -b 200 -o question-output.json -p paragraph-output.csv /path/to/model/directory`

Then the "question-output.json" can be used with the standard triviaqa evaluation [script](https://github.com/mandarjoshi90/triviaqa), 
the "paragraph-output.csv" contains per-paragraph output, we can run  

`python docqa/eval/ranked_triviaqa_scores.py paragraph-output.csv`

to get ranked scores as more paragraphs as used.

### User Input
"docqa/scripts/run_on_user_documents.py" serves as a heavily commented example of how to run our models 
and pre-processing pipeline on other kinds of text. For example:
 
 `docqa/scripts/run_on_user_text.py /path/to/model/directory 
 "Who wrote the satirical essay 'A Modest Proposal'?"  
 ~/data/triviaqa/evidence/wikipedia/A_Modest_Proposal.txt 
 ~/data/triviaqa/evidence/wikipedia/Jonathan_Swift.txt`
 
## Pre-Trained Models
We have four pre-trained models

1. "squad" Our model trained on the standard SQuAD dataset, this model is listed on the SQuAD leaderboard 
as BiDAF + Self Attention

2. "squad-shared-norm" Our model trained on document-level SQuAD using the shared-norm approach. 

3. "triviaqa-web-shared-norm" Our model trained on TriviaQA web with the shared-norm approach. This 
is the model we used to submit scores to the TriviaQA leader board.
 
4. "triviaqa-unfiltered-shared-norm" Our model trained on TriviaQA unfiltered with the shared-norm approach.
This is the model that powers our demo.

The models can be downloaded [here](https://drive.google.com/open?id=1Hj9WBQHVa__bqoD5RIOPu2qDpvfJQwjR)

The models use the cuDNN implementation of GRUs by default, which means they can only be run on
the GPU. We also have slower, but CPU compatible, versions [here](https://drive.google.com/open?id=1NRmb2YilnZOfyKULUnL7gu3HE5nT0sMy).