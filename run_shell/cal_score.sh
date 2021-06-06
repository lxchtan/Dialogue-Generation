export PYTHONPATH=`pwd`

#embeddings bin file: recommend to download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
EMB=../data/GoogleNews-vectors-negative300.bin
PARAMS=config/baseline/GPT2.json
SCORE=results/score_0514.json

calculation(){
  # Calculate Others
  CUDA_VISIBLE_DEVICES=$GPUID python analyse/score.py --embeddings=${EMB} --result_file=${RES} --scorefile=${SCORE}
  # Calculate PPL
  CUDA_VISIBLE_DEVICES=$GPUID python -u evaluator.py --params_file=${PARAMS} \
    --model_checkpoint=${OUTPUT} --score_file=${SCORE} --record_name=${RES} \
    --batch_size 64
}

GPUID=1
OUTPUT=runs/GPT2_7677
RES=results/20210508_GPT2_7677_batch.json
calculation

# GPUID=2
# OUTPUT=runs/GPT2_43
# RES=results/20210508_GPT2_43_batch.json
# calculation