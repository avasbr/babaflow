#!/bin/bash

CUDA_VISIBLE_DEVICES= python dumbnet_distributed_training.py --job_name="ps" --task_index=0
CUDA_VISIBLE_DEVICES=0 python dumbnet_distributed_training.py --job_name="worker" --task_index=0
CUDA_VISIBLE_DEVICES=1 python dumbnet_distributed_training.py --job_name="worker" --task_index=1