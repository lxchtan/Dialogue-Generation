# pipeline: bash run_shell/B0_GPT2.sh 43
# only train: bash run_shell/B0_GPT2.sh 43 train
# only test: bash run_shell/B0_GPT2.sh 43 test

PARAMS=config/baseline/GPT2.json
OUTPUT_PRE=GPT2
if [[ $1 == "7677" ]]; then
  GPUID=3
elif [[ $1 == "43" ]]; then
  GPUID=1
elif [[ $1 == "13" ]]; then
  GPUID=0
elif [[ $1 == "91" ]]; then
  GPUID=2
fi
OUTPUT=${OUTPUT_PRE}_$1

if [[ $2 != "test" ]]; then
  CUDA_VISIBLE_DEVICES=$GPUID python -u trainer.py --params_file ${PARAMS} --output_path $OUTPUT --n_epochs 15 \
    --train_batch_size 8 --valid_batch_size 8 --gradient_accumulation_steps 8 --seed $1
fi

if [[ $2 != "train" ]]; then
  # cal ppl
  CUDA_VISIBLE_DEVICES=$GPUID python -u evaluator.py --params_file ${PARAMS} \
    --model_checkpoint runs/$OUTPUT --result_file results/`date +%Y%m%d`_${OUTPUT}_score.json --record_name `date +%Y%m%d`_${OUTPUT} \
    --batch_size 64

  # generate
  CUDA_VISIBLE_DEVICES=$GPUID python generator.py --params_file ${PARAMS} --generate_config config/generation/generate.json \
    --model_checkpoint runs/$OUTPUT \
    --result_file `date +%Y%m%d`_${OUTPUT}.json --batch_size 64
fi
