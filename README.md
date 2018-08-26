Multi-Domain Neural Machine Translation with Word-Level Domain Context Discrimination
=====================================================================

This codebase contains all scripts except training corpus to reproduce our results in the paper.

### Installation

The following packages are needed:

- Python >= 2.7
- numpy
- Theano >= 0.7 (and its dependencies).

### Preparation

First, preprocess your training corpus. Use BPE(byte-piar-encoding) to segment text into subword units. Please follow <https://github.com/rsennrich/subword-nmt> for further details.

To obtain vocabulary for training, run:

    python scripts/buildvocab.py --corpus /path/ch.train --output /path/to/ch.voc3.pkl \
    --limit 32000 --groundhog
    python scripts/buildvocab.py --corpus /path/en.train --output /path/to/en.voc3.pkl \
    --limit 32000 --groundhog

Similarly, vocabularies for English-French translation can be obtained in the same way.

And also, it's preferred, but not required to initialize encoder-backward decoder component with pretrained parameters in the proposed model of this work.

### Training

For Chinese-English experiment, do the following:

    python -u rnnsearch.py train \
    --corpus /path/ch.train \
	/path/en.train \
	/path/tag.train \
	/path/ch.test \
	/path/en.test \
	/path/tag.test \
    --vocab /path/ch.voc3.pkl /path/en.voc3.pkl \
    --lambda 0.1 \
    --ext-val-script scripts/validate-zhen.sh \
    --model zhen \
    --embdim 500 500 \
    --hidden 1000 1000 1000 1000 300\
    --maxhid 500 \
    --deephid 500 \
    --maxpart 2 \
    --alpha 5e-4 \
    --norm 1.0 \
    --batch 80 \
    --maxepoch 5 \
    --seed 1235 \
    --freq 500 \
    --vfreq 1500 \
    --sfreq 500 \
    --sort 32 \
    --validation /path/ch.dev \
    --references /path/en.dev \
    --optimizer adam \
    --shuffle 1 \
    --keep-prob 0.9 \
    --limit 60 60 60 \
    --delay-val 1 \
    > log.chen 2>&1 &

The training procedure continues about 2 days On a single Nvidia Titan x GPU.


### Evaluation

The evaluation metric we use is case-sensitive BLEU on tokenized reference. Translate the test set and restore text to the original segmentation:

    python rnnsearch.py translate --model ende.best.pkl < /path/to/BPE/newstest2015.en \
    | scripts/restore_bpe.sh > newstest2015.de.trans

And evaluation proceeds by running:

    perl scripts/multi-bleu.perl /path/to/newstest2015.tc.de < newstest2015.de.trans
