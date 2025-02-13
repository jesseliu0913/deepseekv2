#!/bin/bash

mmlu_task="anatomy astronomy business_ethics clinical_knowledge college_biology college_chemistry college_computer_science college_mathematics college_medicine college_physics computer_security conceptual_physics econometrics electrical_engineering elementary_mathematics formal_logic global_facts high_school_biology high_school_chemistry high_school_computer_science high_school_european_history high_school_geography high_school_government_and_politics high_school_macroeconomics high_school_mathematics high_school_microeconomics high_school_physics high_school_psychology high_school_statistics high_school_us_history high_school_world_history human_aging human_sexuality international_law jurisprudence logical_fallacies machine_learning management marketing medical_genetics miscellaneous moral_disputes moral_scenarios nutrition philosophy prehistory professional_accounting professional_law professional_medicine professional_psychology public_relations security_studies sociology us_foreign_policy virology world_religions"

mkdir -p ./log/raw/

for item in $mmlu_task; do
    log_file="./log/raw/mmlu.$item"
    
    echo "Starting evaluation for $item, logging to $log_file"
    CUDA_VISIBLE_DEVICES=4,5 nohup python eval_mt_model.py lukaemon/mmlu "$item" test 1 --sub_one input > "$log_file" 2>&1
    
    echo "Finished evaluation for $item"
done

echo "All tasks have been completed."
