CUDA_VISIBLE_DEVICES=6 nohup python eval_raw_model.py piqa none test 3 --sub_one goal --sub_two sol1 --sub_three sol2 > ./log/raw/piqa.lb 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_raw_model.py winogrande winogrande_debiased test 3 --sub_one sentence --sub_two option1 --sub_three option2 > ./log/raw/winogrande_debiased.lb 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_raw_model.py Rowan/hellaswag none test 3 --sub_one ctx_a --sub_two ctx_b --sub_three activity_label > ./log/raw/hellaswag.lb 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python eval_raw_model.py truthful_qa generation validation 2 --sub_one question --sub_two best_answer > ./log/raw/generation.lb 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_raw_model.py truthful_qa multiple_choice validation 3 --sub_one question --sub_two mc1_targets --sub_three mc2_targets > ./log/raw/multiple_choice.lb 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_raw_model.py gsm8k main test 2 --sub_one question --sub_two answer > ./log/raw/main.lb 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_raw_model.py gsm8k socratic test 2 --sub_one question --sub_two answer > ./log/raw/socratic.lb 2>&1 &

