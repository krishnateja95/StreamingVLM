#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle:grand
#PBS -l walltime=20:00:00
#PBS -q preemptable
#PBS -A datascience

module use /soft/modulefiles/
module load conda
conda activate streamingVLM

export LMUData="/lus/grand/projects/datascience/krishnat/home_dir_code/datasets/"

cd /lus/grand/projects/datascience/krishnat/home_dir_code/Active_projects/StreamingVLM/VLMEvalKit
cache_dir='/lus/grand/projects/datascience/krishnat/model_weights/LLaMA/llama_cache/'

work_dir="./phi3_5v_baseline"
python run.py --work-dir $work_dir --data MME --model Phi-3.5-Vision --mode all --saveresults --cache_dir=$cache_dir

work_dir="./molmo_baseline"
python run.py --work-dir $work_dir --data MME --model molmoE-1B-0924 --mode all --saveresults --cache_dir=$cache_dir 

work_dir="./Qwen2_VL_2B_baseline"
python run.py --work-dir $work_dir --data MME --model Qwen2-VL-2B-Instruct --mode all --saveresults --cache_dir=$cache_dir 

