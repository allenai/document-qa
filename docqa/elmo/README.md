## ELMo
This contains the (pretty rough) code for running our SQuAD model with ELMo weights

To train or test the model you need the pre-trained ELMo model. It can be downloaded 
[here](https://docs.google.com/uc?export=download&id=1vXsiRHxJqsj3HLesUIet0x4Yrjw0S54D).
Then unzip it and store in ~/data/lm (or change config.py to alter its expected location). For example:

```
mkdir -p ~/data/lm
cd ~/data/lm 
mv ~/Download/squad-context-concat-skip.tar.gz .
tar -xzf squad-context-concat-skip.tar.gz
rm squad-context-concat-skip.tar.gz
```

### Training
Now the model can be trained using:

`python docqa/elmo/ablate_elmo_model.py`

### Testing
The model can be tested on the dev set using our standard evaluation script:

`python docqa/eval/squad_eval.py -o output.json -c dev /path/to/model/directory`

Note by default the language model will use word vectors
that were pre-computed for the SQuAD corpus. Running it on
other kinds of data takes a bit more work, 
see "docqa/elmo/run_on_user_text.py" for an example with comments. 
Using the script we can run the model on user-defined input, for example:

`docqa/elmo/run_on_user_text.py /path/to/model/directory "What color are apples" "Apples are blue"`


### Pre-Trained Model
The pre-trained model we used for SQuAD can be downloaded [here](https://drive.google.com/open?id=1GuKh2TJFF6FIhiFpoFslJ1WPlGAxAISt)


### Codalab
The codalab worksheet we use to get our SQuAD test scores is 
[here](https://worksheets.codalab.org/worksheets/0xc7fd7c36337146838b9b064a327e59fd/).
