#!/bin/bash

#if [[ $# -lt 1 ]] || [[ $# -gt 1 ]]; then
#    echo "Incorrect number of arguments"
#    echo "Usage: $0 config_file"
#    exit 1
#fi

CONFIG_FILE=${1}


edir="${DUMP_DIR}"
ename="${JOB_ID}_v${MAST_HPC_JOB_VERSION}_a${MAST_HPC_JOB_ATTEMPT_INDEX}"
dataset_path="/mnt/wsfuse/mlperf_llama"


echo "wenyin: start"
echo dump_dir=$edir
echo experiment_name=$ename



LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so.525.105.17"
export LIBCUDA_DIR="${LIBCUDA%/*}"
export LD_PRELOAD="${PRELOAD_PATH:=$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so.525.105.17}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_DIR}/lib"
export PYTHONPATH="$PYTHONPATH:$TORCHX_RUN_PYTHONPATH"

source ${CONDA_DIR}/bin/activate

cd /packages/infra_mlperf_llama


###############
#  do whatever you like below
###############

#python scripts/train.py --max_seq_len 8192 --bf16 True --logging_steps 1 --eval_steps 22 --output_dir "/tmp/llama-70b" --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --lr_scheduler_type "cosine" --learning_rate 1e-3 --warmup_ratio 0.03 --use_gradient_checkpointing True --use_peft_lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 --max_steps 1 --use_flash_attn --lora_target_modules "q_proj,v_proj,k_proj,o_proj" 

#sleep 1h 

python lora_finetune_distributed.py --config recipes/configs/llama2/70B_lora.yaml

#TORCH_DISABLE_ADDR2LINE=1  \
#python train.py  \
#--job.config_file "${CONFIG_FILE}" \
#--job.dump_folder "${edir}" \
#--training.dataset_path "${dataset_path}"
