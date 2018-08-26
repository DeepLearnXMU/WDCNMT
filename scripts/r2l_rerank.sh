src=$1
model_l2r=$2
model_r2l=$3
beamsize=50

cat $src | python rnnsearch.py translate --model $model_l2r --beamsize $beamsize --n-best --normalize | python scripts/reverse_nbest.py | python rnnsearch.py rescore --model $model_r2l --source $src --normalize | python scripts/rerank.py | scripts/reverse.sh