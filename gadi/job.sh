#!/bin/bash
#PBS -P hn98
#PBS -l storage=gdata/hn98

# iterating through LR
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=2,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.001,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=2,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0001,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=2,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120

# iterating through feature count
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=5,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=4,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=3,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=1,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120

# graph norm off
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=2,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=0 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120

# larger cluster radius
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=2,subgraph_size=64,radius=0.1,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120

# larger subgraphs
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=2,subgraph_size=1000,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120

# less hidden layers (64)
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=1,seed=$i,features=2,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1,hidden=64 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120

# baseline GWN
for i in $(seq 1 10); do qsub -v dataset=PEMSBAY,pretrain=0,seed=$i,features=2,subgraph_size=64,radius=0.01,pretrn_epoch=50,epoch=100,network_calls=0,lr=0.0003,graph_norm=1,hidden=64 /g/data/hn98/matt/job-scpt.sh; sleep 60; done
sleep 120
