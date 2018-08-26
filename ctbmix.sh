#!/bin/bash
model="DSA-3"
gpunum="gpu1"
pro=".iter4-0.pkl"

export THEANO_FLAGS=device=${gpunum}

nohup python -u rnnsearch.py train --model ${model} \
--corpus /home/lemon/data/UM-Corpus/domain4/domain4ch.train \
/home/lemon/data/UM-Corpus/domain4/domain4en.train \
/home/lemon/data/UM-Corpus/domain4/domain4tag.train \
/home/lemon/data/UM-Corpus/domain4/domain4ch.test \
/home/lemon/data/UM-Corpus/domain4/domain4en.test \
/home/lemon/data/UM-Corpus/domain4/domain4tag.test \
--validation /home/lemon/data/UM-Corpus/domain4/domain4ch.dev \
--references /home/lemon/data/UM-Corpus/domain4/domain4en.dev \
--vocab /home/lemon/data/UM-Corpus/domain4/domain4ch.vocab.pkl \
/home/lemon/data/UM-Corpus/domain4/domain4en.vocab.pkl \
--lambda 0.1 \
--config trainin.settings.yaml > ${model}.train 2>&1 &
#nohup bash goontrain.sh $model $pro $gpunum &
train_id=$!
wait $train_id

a=(Laws News Spoken Thesis)
for i in ${a[@]}
do
    nohup python rnnsearch.py translate --model ${model}.best.pkl --normalize < \
    ~/data/UM-Corpus/domain4/${i}.ch.test.BPE 1>${model}${i}.bpe 2>${model}${i}.out &
    trans_id=$1
    wait $trans_id
done

for i in ${a[@]}
do
    cat ${model}${i}.bpe | sed -r 's/(@@ )|(@@ ?$)//g' | cat >${model}${i}.txt
    nohup perl scripts/multi-bleu.perl -lc ~/data/UM-Corpus/domain4/${i}.en.test < \
    ${model}${i}.txt > ${model}${i}.bleu 2>&1 &
    bleu_id=$!
    wait $bleu_id
done

cat ${model}Laws.txt ${model}News.txt ${model}Spoken.txt ${model}Thesis.txt > ${model}.txt
nohup perl scripts/multi-bleu.perl -lc ~/data/UM-Corpus/domain4/domain4en.test < \
    ${model}.txt > ${model}.bleu 2>&1 &
