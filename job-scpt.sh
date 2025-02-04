#!/bin/bash

#PBS -P hn98
#PBS -q gpuvolta
#PBS -l walltime=06:00:00
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l storage=gdata/hn98

# for network -q cpuvolta, ngpus=0, ncpus=1, mem=4GB

module load python3/3.9.2
module list

source /g/data/hn98/matt/venv/bin/activate

pip list

cd /g/data/hn98/matt/forecasting-on-new-roads
pwd
nvidia-smi
lscpu

# 0: .py file
# 1: IS_PRETRN
# 2: R_TRN
# 3: IS_EPOCH_1
# 4: seed
# 5: TEMPERATURE
# 6: dataset
# 7: seed_ss # spatial split
# 8: IS_DESEASONED
# 9: weight_decay
# 10: adp_adj
# 11: is_SGA

python3 ./pred_GWN_16_adpAdj.py \
${pretrain} `# 1: IS_PRETRN` \
0.7 `# 2: R_TRN` \
0 `# 3: IS_EPOCH_1` \
${seed} `# 4: seed` \
100 `# 5: TEMPERATURE`\
${dataset} `# 6: dataset` \
${seed} `# 7: seed_ss  spatial split` \
${pretrain} `# 8: IS_DESEASONED` \
0.0 `# 9: weight_decay` \
${pretrain} `# 10: adp_adj` \
${pretrain} `# 11: is_SGA` \
${features} `# 12: num_features` \
${subgraph_size} \
${radius} \
${pretrn_epoch} \
${epoch} \
${network_calls} \
${lr} \
${graph_norm} \
${hidden}
