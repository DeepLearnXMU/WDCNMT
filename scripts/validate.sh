#!/bin/bash
set -e

entry=$1
model=$2
src=$3
ref_stem=$4

beamsize=10
bpe=false
bleu_script=/home/pluiefox/repos/mosesdecoder/scripts/generic/multi-bleu.perl

translate="python $entry translate --model $model --beamsize $beamsize --normalize"
calc_bleu="perl $bleu_script -lc $ref_stem"

if [[ $bpe == "true" ]]; then
    bleu=$($translate < $src | sed -r 's/(@@ )|(@@ ?$)//g' | $calc_bleu | cut -f 3 -d ' ' | cut -f 1 -d ',')
else
    bleu=$($translate < $src | $calc_bleu | cut -f 3 -d ' ' | cut -f 1 -d ',')
fi

echo $bleu
