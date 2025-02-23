from visualization import load_subset_experiment, calculate_test_performance
from data_object import DataObject, DataObjectConstants
from folder_names import FolderNames
from plotting import Plotting
from models import Models
import numpy as np
import traceback
import argparse
import pickle
import torch
import os

# import wandb
# wandb.login(anonymous='allow', key='')

def main(model_names, existing_data_name, new_data_name, threshold, subset_percentage):
    # run = wandb.init(
    #     project="Optimizing Data Selection",
    # )

    # all experimental configurations
    uc_labels = ["Initial", "DEFT UCS", "LESS", "Model Dependent + CG FL", "Model Dependent + FL Only", "SelectIT", "Model Independent + CG FL", "Random", "Full Dataset"]
    ucl_shorthand = ["initial", "deft_ucs", "less", "mod_dep_fl", "mod_dep_flonly", "select_it", "mod_ind_fl", "random", "full_data"]
    # sl_labels = ["PEFT"]
    sl_labels = ["ICL", "PEFT"]

    if "125m" in model_names[0]:
        sl_labels.append('FFT')

    # loop through each of the model names
    for model_name in model_names:
        if existing_data_name == new_data_name:
            fn = FolderNames(model_name, "same_data_cache")
            labels = ["P3", "mix-instruct", "natural-instructions"]
        elif "benchmark" in new_data_name:
            fn = FolderNames(model_name, "benchmark_cache")
            labels = ["benchmark_gsm8k", "mix-instruct"]
            # labels = ["benchmark_mmlu", "benchmark_mt_bench_human_judgments"]
        else:
            fn = FolderNames(model_name, "version_cache")
            labels = ["gov", "ibm_ft", "hotpot_qa", "squad"]

        models = Models(language_model_name=model_name)

        with open(fn.visualization_cache_file, 'rb') as f:
            vis_dims, all_data = pickle.load(f)
        
        existing_data_ind = labels.index(existing_data_name)
        new_data_ind = labels.index(new_data_name)

        # set up training and validation sets for the DataObject instance
        num_exist_train, num_new_train = len(all_data[existing_data_ind][0]), len(all_data[new_data_ind][0])
        num_exist_valid, num_new_valid = len(all_data[existing_data_ind][1]), len(all_data[new_data_ind][1])
        exist_point_labels = [np.array([f"{existing_data_ind}-{i}" for i in range(len(all_data[existing_data_ind][0]))]), 
                            np.array([f"{existing_data_ind}-{num_exist_train+i}" for i in range(len(all_data[existing_data_ind][1]))]),
                            np.array([f"{existing_data_ind}-{num_exist_train+num_exist_valid+i}" for i in range(len(all_data[existing_data_ind][2]))]),]
        new_point_labels = [np.array([f"{new_data_ind}-{i}" for i in range(len(all_data[new_data_ind][0]))]), 
                            np.array([f"{new_data_ind}-{num_new_train+i}" for i in range(len(all_data[new_data_ind][1]))]),
                            np.array([f"{new_data_ind}-{num_new_train+num_new_valid+i}" for i in range(len(all_data[new_data_ind][2]))])]
    
        # create a DataObject instance
        if existing_data_name == new_data_name:
            data = DataObject([existing_data_name], [existing_data_ind], [new_data_name], [new_data_ind], [all_data[existing_data_ind][0]], [vis_dims[existing_data_ind][0]], [exist_point_labels[0]],
                        [all_data[new_data_ind][1]], [vis_dims[new_data_ind][1]], [new_point_labels[1]],
                        case=DataObjectConstants.DATA_OBJECT_SAME_DATSET)
        elif "benchmark" in new_data_name:
            data = DataObject(existing_data_name, existing_data_ind, new_data_name, new_data_ind, all_data[existing_data_ind], vis_dims[existing_data_ind], exist_point_labels,
                        all_data[new_data_ind], vis_dims[new_data_ind], new_point_labels,
                        case=DataObjectConstants.DATA_OBJECT_BENCHMARK)
        else:
            data = DataObject(existing_data_name, existing_data_ind, new_data_name, new_data_ind, all_data[existing_data_ind], vis_dims[existing_data_ind], exist_point_labels,
                        all_data[new_data_ind], vis_dims[new_data_ind], new_point_labels,
                        case=DataObjectConstants.DATA_OBJECT_NEW_VERSION)
        
        # define the dataset configuration code (a code that indicates the combination of datasets one is using)
        dataset_config_code = fn.dataset_config_file_code(existing_data_name, new_data_name)
        data.set_dataset_config_code(dataset_config_code)

        # create a Plotting instance
        plotting = Plotting(data, labels, models, fn)

        # loop through all combinations of experiments
        for subset_learning in reversed(sl_labels):
            for utility_criteria in reversed(uc_labels):
                if "initial" in utility_criteria:
                    exp_config = utility_criteria + "-" + subset_learning + "-" + str(subset_percentage)
                else:
                    exp_config = ucl_shorthand[uc_labels.index(utility_criteria)] + "-" + subset_learning + "-" + str(subset_percentage)

                try:
                    # if not os.path.exists(fn.exp_knowledge_file(dataset_config_code, exp_config)):
                    #     continue
                    load_subset_experiment(existing_data_name, existing_data_ind, new_data_name, new_data_ind, exp_config, utility_criteria, subset_learning, 
                                    subset_percentage, threshold, labels, data, plotting, models, fn)
                
                    rouge_val, _ = calculate_test_performance(all_data[new_data_ind][1], data, exp_config, models, fn, score="rouge")
                    bge_val, _ = calculate_test_performance(all_data[new_data_ind][1], data, exp_config, models, fn, score="bge")
                    llmaj_val, _ = calculate_test_performance(all_data[new_data_ind][2], data, exp_config, models, fn, score="prometheus")
                except:
                    hi = 9
                # my_table = wandb.Table(columns=['ROUGE', 'BGE', 'LLM-A-J'])
                # my_table.add_data(rouge_val[0], bge_val, llmaj_val)
                # run.log({f"{data.use_case} - {model_name}, {exp_config}": my_table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--subset_percentage", type=float, default=0.3)
    parser.add_argument("--existing_data_name", type=str, default="mix-instruct")
    parser.add_argument("--new_data_name", type=str, default="mix-instruct")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m") #mistralai/Mistral-7B-v0.1") #meta-llama/Llama-3.2-3B")
    args = parser.parse_args()

    main([args.model_name], args.existing_data_name, args.new_data_name, args.threshold, args.subset_percentage)

