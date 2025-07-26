#!/bin/bash
EVAL_FILE=/pubshare/fwk/code/SeRL/evaluation/Health/outputs/home/jovyan/share/LLMAgent/model/Llama-3.2-3B-Instruct/med_qa/medical_qa_medical_qa_-1_seed0_t0.6_s0_e-1.jsonl

python evaluate.py $EVAL_FILE