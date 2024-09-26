for i in $(seq 1 10);
do
    echo $i

    MODEL_NAME='Qwen/Qwen2-72B-Instruct'
    python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME --subset_percentage=0.05
    python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME --subset_percentage=0.15
    python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME --subset_percentage=0.50
    python3 visualization/load_all_experiments.py --existing_data_name mix-instruct --new_data_name mix-instruct --model_name=$MODEL_NAME --subset_percentage=0.99

    python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME --subset_percentage=0.05
    python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME --subset_percentage=0.15
    python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME --subset_percentage=0.50
    python3 visualization/load_all_experiments.py --existing_data_name ibm_ft --new_data_name gov --model_name=$MODEL_NAME --subset_percentage=0.99

done

notify "experiments done - graph"