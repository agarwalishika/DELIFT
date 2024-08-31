from data_object import DataObject, DataObjectConstants, get_prompts_refs
from transformers import AutoModelForCausalLM
from folder_names import FolderNames
from plotting import Plotting
from lm_knowledge import *
from models import Models
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import csv
import os

import sys
sys.path.append('.')
from subset_selection.inference_icl import InferenceICL

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_input_output(data: DataObject, inds, labels, exist_ind, generated_texts=None):
    """
    Used to obtain the inputs and outputs for data selections on any graph, which users can then download to .tsv files.

    Args:
        data: instance of DataObject from data_object.py
        inds: indices of the points in the selection
        labels: list of dataset names present in the cache/dataset_pkls
        exist_ind: index of the existing dataset in the `labels` list
        generated_texts: if provided, the corresponding predicted outputs from the model will also be sorted through
    """
    input, output, gen_text = [], [], []
    for i, l in zip(inds, labels):

        if int(l.split('-')[0]) == exist_ind:
            # point is from the existing dataset
            num_train = len(data.train_existing_prompts)
            num_valid = len(data.valid_existing_prompts)
            if i >= num_train + num_valid:
                # point is in the testing set
                inp = data.test_existing_prompts[i-num_train-num_valid-1]
                out = data.test_existing_references[i-num_train-num_valid-1]
            elif i >= num_train:
                # point is in the validation set
                inp = data.valid_existing_prompts[i-num_train-1]
                out = data.valid_existing_references[i-num_train-1]
                if generated_texts: # validation set is only visualized
                    gen_text.append(generated_texts[i-num_train])
            else:
                # point is in the training set
                inp = data.train_existing_prompts[i]
                out = data.train_existing_references[i]
        
        else:
            # point is from the new dataset
            num_train = len(data.train_new_prompts)
            num_valid = len(data.valid_new_prompts)
            if i >= num_train + num_valid:
                # point is in the testing set
                inp = data.test_new_prompts[i-num_train-num_valid-1]
                out = data.test_new_references[i-num_train-num_valid-1]
            elif i >= num_train:
                # point is in the validation set
                inp = data.valid_new_prompts[i-num_train-1]
                out = data.valid_new_references[i-num_train-1]
                if generated_texts: # validation set is only visualized
                    gen_text.append(generated_texts[i-num_train])
            else:
                # point is in the training set
                inp = data.train_new_prompts[i]
                out = data.train_new_references[i]
        
        # parse through the input and output
        input.append(inp[inp.index("Input:")+6:].strip())
        output.append(out.strip())

    if generated_texts:
        return input, output, gen_text
    else:
        return input, output


def get_info(points, inds=[], point_nums=[], labels=[]):
    """
    Used to process the points in the data selection.

    Args:
        points: st.session_state.GRAPH_NAME.selection.points object
        inds: [optional] if provided, only unique indices will be added
        point_nums: [optional] if provided, only unique point numbers will be added
    """
    for point in points:
        l = point['text']
        if l not in point_nums:
            inds.append(int(point['text'].split('-')[-1]))
            point_nums.append(l)
            labels.append(int(point['curve_number']))
    return inds, point_nums, labels

def display_selection(container, state, data: DataObject, exist_ind, label_names, gen_texts=None, suffix=""):
    """
    Used to create the .tsv file to download for viewing the selection data on a graph.

    Args:
        container: st.container object in which the download button will pop up
        state: st.session_state.GRAPH_NAME object
        data: instance of DataObject from data_object.py
        exist_ind: index of the existing data in `label_names`
        label_names: array of names of datasets from the cache/dataset_pkls
        gen_texts: array of generated texts that can also be part of the .tsv file (mostly viable on the BEFORE/AFTER experiment graphs as they display the validation set, and model inference is performed on validation sets)
        suffix
    """
    inds, point_nums, labels = get_info(state.selection.points)
    temp = get_input_output(data, inds, point_nums, exist_ind, generated_texts=gen_texts)
    if gen_texts:
        input, output, selected_gens = temp
    else:
        input, output = temp

    if len(point_nums) <= 0:
        return

    point_labels = []
    for label in labels:
        point_labels.append(label_names[int(label)])

    if gen_texts:
        header = ["point number", "input", "output", "predicted_output", "label"]
    else:
        header = ["point number", "input", "output", "label"]
    file_name = f"selection{suffix}.tsv"

    with open(file_name, 'w+') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        tsv_writer.writerow(header)
        if gen_texts:
            for l, i, o, g, t in zip(point_nums, input, output, selected_gens, point_labels):
                tsv_writer.writerow([l, i, o, g, t])
        else:
            for l, i, o, t in zip(point_nums, input, output, point_labels):
                tsv_writer.writerow([l, i, o, t])

    with container:
        with open(file_name, 'r') as tsv_file:
            st.download_button('Download selection data', tsv_file, f"selection{suffix}.tsv")

def get_label_names_from_fig(fig):
    """
    Extract the labels from the legend on the given Plotly Graph Object.

    Args:
        fig: instance of a Plotly Graph Object
    """
    label_names = []
    for data in fig.data:
        label_names.append(data.name)
    return label_names

def calculate_test_performance(test_data, data: DataObject, exp_config, models: Models, fn: FolderNames, score):
    """
    Performs inference on the test set and returns an overall performance.

    Args:
        test_data: list of strings that contain both prompt and reference
        data: instance of DataObject from data_object.py
        exp_config: shorthand code that explains the experiment configuration
        models: instance of Models from models.py
        fn: instance of FolderNames from folder_names.py
        score: "bleu" or "rouge" that indicates the performance metric to use
    """
    test_prompts, test_references = get_prompts_refs(test_data)
    exp_file = fn.exp_knowledge_file(data.dataset_config_code, exp_config, prefix="test-")

    # check if the inference results (only the generated text) was stored already
    if os.path.exists(exp_file):
        with open(exp_file, 'rb') as f:
            generated_text = pickle.load(f)
        return calculate_similarity(generated_text, test_references, score=score, return_invidiual=False), generated_text

    # if ICL is the subset quality evaluation technique
    if "ICL" in exp_config:
        # create the ICL example-augmented prompts if they aren't already stored
        icl = InferenceICL(models.embedding_model, models.embedding_tokenizer)
        icl_io_file = fn.icl_io_file(data.dataset_config_code, exp_config, prefix="test-")
        if not os.path.exists(icl_io_file):
            icl_prompts, icl_references = icl.create_icl_inference_data(data.train_new_data_sub, test_prompts, test_references)
            with open(icl_io_file, 'wb+') as f:
                pickle.dump((icl_prompts, icl_references), f)
        else:
            with open(icl_io_file, 'rb') as f:
                icl_prompts, icl_references = pickle.load(f)
        # perform inference
        metrics, generated_text = perform_inference(models.language_model, models.language_tokenizer, icl_prompts, icl_references)
    
    # if PEFT is the subset quality evaluation technique
    if "PEFT" in exp_config:
        peft_model_dir = fn.peft_ft_model(data.dataset_config_code, exp_config)
        lora_model = AutoModelForCausalLM.from_pretrained(peft_model_dir).to(device)
        metrics, generated_text = perform_inference(lora_model, models.language_tokenizer, test_prompts, test_references)
    
    with open(exp_file, 'wb+') as f:
        pickle.dump(generated_text, f)
    return metrics, generated_text

def visualize_subset_experiment(existing_data_name, exist_ind, new_data_name, new_ind, exp_config, utility_criteria, subset_learning, subset_percentage, knowledge_threshold, labels, data: DataObject, plotting: Plotting, models: Models, fn: FolderNames):
    """
    Used for end-to-end visualization of one subset experiment (from running the experiment, to storing, and displaying the graphs)

    Args:
        existing_data_name: name of the EXISTING dataset
        exist_ind: index of the EXISTING dataset wrt the labels array
        new_data_name: name of the NEW dataset
        new_ind: index of the NEW dataset wrt the labels array
        exp_config: shorthand code that explains the experiment configuration
        utility_critera: name of utility (model dependent ICL utility/gradient utility or model independent)
        subset_learning: name of subset quality evaluation (ICL or PEFT)
        subset_percentage: percentage of the training set to choose in our subset
        knowledge_threshold: number between 0 and 1 that indicates good/bad performance
        labels: list of dataset names present in the cache/dataset_pkls
        data: instance of DataObject in data_object.py
        plotting: instance of Plotting from plotting.py
        models: instance of Models from models.py
        fn: instance of FolderNames from folder_names.py
    """
    before_fig, subset_fig, after_fig = load_subset_experiment(existing_data_name, exist_ind, new_data_name, new_ind, exp_config, utility_criteria, subset_learning, subset_percentage, knowledge_threshold, labels, data, plotting, models, fn)

    # display experimental results
    graphs, tables = st.container(), st.container()

    def display_before_selection():
        with open(fn.exp_knowledge_file(data.dataset_config_code, "before_exp-" + exp_config), 'rb') as f:
            gen_text = pickle.load(f)
        display_selection(tables, st.session_state.before, data, exist_ind, label_names=get_label_names_from_fig(before_fig), gen_texts=gen_text, suffix="_before")
    def display_after_selection():
        with open(fn.exp_knowledge_file(data.dataset_config_code, exp_config), 'rb') as f:
            gen_text = pickle.load(f)
        display_selection(tables, st.session_state.after, data, exist_ind, label_names=get_label_names_from_fig(after_fig), gen_texts=gen_text, suffix="_after")

    with graphs:
        col1, col2 = st.columns(2, vertical_alignment="top")

        # BEFORE: What does the LM "know"?
        with col1:
            st.plotly_chart(before_fig, on_select=display_before_selection, key='before')

        # AFTER: what does the LM "know"?
        with col2:
            st.plotly_chart(after_fig, on_select=display_after_selection, key='after')
        
    # display the subset
    sub_graph, sub_table = st.container(), st.container()
    def display_subset_selection():
        display_selection(sub_table, st.session_state.subset, data, exist_ind, label_names=get_label_names_from_fig(subset_fig), suffix="_subset")
    with sub_graph:
        st.plotly_chart(subset_fig, on_select=display_subset_selection, key='subset')


def load_subset_experiment(existing_data_name, exist_ind, new_data_name, new_ind, exp_config, utility_criteria, subset_learning, subset_percentage, knowledge_threshold, labels, data: DataObject, plotting: Plotting, models: Models, fn: FolderNames):   
    """
    Used for running the experiment and storing the results

    Args:
        existing_data_name: name of the EXISTING dataset
        exist_ind: index of the EXISTING dataset wrt the labels array
        new_data_name: name of the NEW dataset
        new_ind: index of the NEW dataset wrt the labels array
        exp_config: shorthand code that explains the experiment configuration
        utility_critera: name of utility (model dependent ICL utility/gradient utility or model independent)
        subset_learning: name of subset quality evaluation (ICL or PEFT)
        subset_percentage: percentage of the training set to choose in our subset
        knowledge_threshold: number between 0 and 1 that indicates good/bad performance
        labels: list of dataset names present in the cache/dataset_pkls
        data: instance of DataObject in data_object.py
        plotting: instance of Plotting from plotting.py
        models: instance of Models from models.py
        fn: instance of FolderNames from folder_names.py
    """

    # BEFORE: what does the LM "know"?
    # icl/peft sets contain the existing training data
    # evaluate on the new validation data
    data.set_icl_peft_sets(data.train_existing_data, data.train_existing_prompts, data.train_existing_references, data.valid_new_prompts, data.valid_new_references)
    before_fig, _, _ = plotting.obtain_experiment_results(new_ind, data, data.dataset_config_code, "before_exp-" + exp_config, prefix="BEFORE", threshold=knowledge_threshold)

    # create subset on the new training data
    subset_fig, subset_idx = plotting.visualize_subset(subset_percentage, utility_criteria, data)
    data.create_train_subset(subset_idx)

    # AFTER: what does the LM "know"?
    # icl/peft sets contain the entire existing training data + subset from the new training data
    # evaluate on the new validation data
    data.set_icl_peft_sets(data.train_new_data_sub, data.train_new_prompts_sub, data.train_new_references_sub, data.valid_new_prompts, data.valid_new_references)
    after_fig, _, _ = plotting.obtain_experiment_results(new_ind, data, data.dataset_config_code, exp_config, prefix="AFTER", threshold=knowledge_threshold)

    return before_fig, subset_fig, after_fig


def correct_value(val):
    """
    Serves as an error checking for user inputs. Keeps percentages in [0, 0.99].

    Args:
        val: user input percentage
    """
    return min(max(val, 0.0), 0.99)

def main():
    st.set_page_config(layout="wide")
    tab1, tab2, tab3 = st.tabs(["Subset Selection", "Data Analysis", "Data Point Viewer"]) #, "About"])

    new_data_name = ""
    existing_data_name = ""

    with tab1:
        # Define the model, datasets and performance thresholds
        col1, col2 = st.columns(2, vertical_alignment="center")
        with col1:
            model_name = st.text_input("Input HuggingFace Model name (after 'huggingface.co/'):", "microsoft/Phi-3-mini-4k-instruct") #"EleutherAI/gpt-neo-125m") #"ibm-granite/granite-7b-base")

            col1a, col1b = st.columns(2, vertical_alignment="center")
            with col1a:
                threshold = correct_value(float(st.text_input("LM performance threshold (in [0, 1]):", "0.7")))
            with col1b:
                subset_percentage = correct_value(float(st.text_input("Size of subset (percentage of training set) (in [0, 1]):", "0.3")))
        
        # Set up data variables for general experiments
        fn = FolderNames(model_name)
        models = Models(language_model_name=model_name)
        labels = [label.split('.')[0] for label in os.listdir(fn.dataset_pkl_folder) if 'all_data' not in label]

        with col2:
            existing_data_name = st.multiselect("Select existing dataset(s)", labels, default=labels[-1])[0]
            new_data_name = st.multiselect("Select dataset(s) to add in", labels, default=labels[0])[0]

        # Set up data variables for the experiments
        with open(fn.visualization_cache_file, 'rb') as f:
            vis_dims, all_data = pickle.load(f)
        existing_data_ind = labels.index(existing_data_name)
        new_data_ind = labels.index(new_data_name)

        num_exist_train, num_new_train = len(all_data[existing_data_ind][0]), len(all_data[new_data_ind][0])
        num_exist_valid, num_new_valid = len(all_data[existing_data_ind][1]), len(all_data[new_data_ind][1])
        exist_point_labels = [np.array([f"{existing_data_ind}-{i}" for i in range(len(all_data[existing_data_ind][0]))]), 
                            np.array([f"{existing_data_ind}-{num_exist_train+i}" for i in range(len(all_data[existing_data_ind][1]))]),
                            np.array([f"{existing_data_ind}-{num_exist_train+num_exist_valid+i}" for i in range(len(all_data[existing_data_ind][2]))]),]
        new_point_labels = [np.array([f"{new_data_ind}-{i}" for i in range(len(all_data[new_data_ind][0]))]), 
                            np.array([f"{new_data_ind}-{num_new_train+i}" for i in range(len(all_data[new_data_ind][1]))]),
                            np.array([f"{new_data_ind}-{num_new_train+num_new_valid+i}" for i in range(len(all_data[new_data_ind][2]))])]
        if existing_data_name == new_data_name:
            data = DataObject(existing_data_name, existing_data_ind, new_data_name, new_data_ind, all_data[existing_data_ind][0], vis_dims[existing_data_ind][0], exist_point_labels[0],
                        all_data[new_data_ind][1], vis_dims[new_data_ind][1], new_point_labels[1],
                        case=DataObjectConstants.DATA_OBJECT_SAME_DATSET)
        elif "benchmark" in new_data_name:
            data = DataObject(existing_data_name, existing_data_ind, new_data_name, new_data_ind, all_data[existing_data_ind], vis_dims[existing_data_ind], exist_point_labels,
                        all_data[new_data_ind], vis_dims[new_data_ind], new_point_labels,
                        case=DataObjectConstants.DATA_OBJECT_BENCHMARK)
        else:
            data = DataObject(existing_data_name, existing_data_ind, new_data_name, new_data_ind, all_data[existing_data_ind], vis_dims[existing_data_ind], exist_point_labels,
                        all_data[new_data_ind], vis_dims[new_data_ind], new_point_labels,
                        case=DataObjectConstants.DATA_OBJECT_NEW_VERSION)
        dataset_config_code = fn.dataset_config_file_code(existing_data_name, new_data_name)
        data.set_dataset_config_code(dataset_config_code)
        plotting = Plotting(data, labels, models, fn)
            
        # create separate tabs for visualizing experiments and viewing results
        graph_visualizer, results_visualizer = st.tabs(["Graph Visualization", "Results"])

        # set some constants
        # uc_labels = ["Initial", "Model Dependent Utility", "Model Independent", "Random", "Full Dataset"]
        # ucl_shorthand = ["before-exp", "mod_dep_fl", "mod_ind_fl", "random", "full_data"]
        uc_labels = ["Initial", "Model Dependent + CG FL", "Model Independent + CG FL", "SelectIT", "Random", "Full Dataset"]
        ucl_shorthand = ["before-exp", "mod_dep_fl", "mod_ind_fl", "select_it", "random", "full_data"] 
        sl_labels = ["ICL", "PEFT"]

        # visualize one experiment
        with graph_visualizer:
            col1, col2 = st.columns(2, vertical_alignment="center")
            with col1:
                ## define utility
                utility_criteria = st.selectbox("Define utility using: ", uc_labels[1:])
            with col2:
                ## choose subset learning technique
                subset_learning = st.selectbox("Learn on the subset using:", sl_labels)

            ## check if the experiment has been loaded
            exp_config = ucl_shorthand[uc_labels.index(utility_criteria)] + "-" + subset_learning + "-" + str(subset_percentage)

            visualize_subset_experiment(existing_data_name, existing_data_ind, new_data_name, new_data_ind, exp_config, utility_criteria, subset_learning, subset_percentage, threshold, labels, data, plotting, models, fn)
        
        # sub-tab to display the tables of results from each experiment configuration
        with results_visualizer:
            rouge_results = {}
            bleu_results = {}

            for utility_criteria in uc_labels:
                temp_rouge = []
                temp_bleu = []
                for subset_learning in sl_labels:
                    # load experiment configuration
                    exp_config = ucl_shorthand[uc_labels.index(utility_criteria)] + "-" + subset_learning + "-" + str(subset_percentage)
                    _, subset_idx = plotting.visualize_subset(subset_percentage, utility_criteria, data)
                    data.create_train_subset(subset_idx)

                    # calculate rouge on test set
                    r_val, _ = calculate_test_performance(all_data[new_data_ind][2], data, exp_config, models, fn, score="rouge")
                    temp_rouge.append(round(100 * r_val[0], 2))

                    # calculate bleu on test set
                    b_val, _ = calculate_test_performance(all_data[new_data_ind][2], data, exp_config, models, fn, score="bleu")
                    temp_bleu.append(round(100 * b_val[0], 2))
                rouge_results[utility_criteria] = temp_rouge
                bleu_results[utility_criteria] = temp_bleu
            
            st.write('Test set ROUGE:')
            df_r = pd.DataFrame.from_dict(rouge_results)
            df_r.columns = uc_labels
            df_r.index = sl_labels
            st.table(df_r)

            st.write('Test set BLEU:')
            df = pd.DataFrame.from_dict(bleu_results)
            df.columns = uc_labels
            df.index = sl_labels
            st.table(df)


    # tab to display the conditional gain and mutual information between the two datasets
    with tab2:
        ## Conditional Gain / MI
        uc_labels = ["Model Dependent Utility", "Model Independent Semantic Similarity"]
        uc_col, perc_col = st.columns(2)
        with uc_col:
            utility_criteria = st.selectbox("Select utility criteria: ", uc_labels)
        with perc_col:
            k_perc = float(st.text_input("Subset portion (in [0, 1]):", "0.3"))

        if "Dependent" in utility_criteria:
            utility_file = fn.model_dep_utility_file(data.dataset_config_code)
        if "Independent" in utility_criteria:
            utility_file = fn.model_ind_utility_file(data.dataset_config_code)
        
        cgmi_fig, _ = plotting.gen_cg_fig(data, utility_file, k_perc)

        cgmi_graph, cgmi_table = st.container(), st.container()
        def display_cgmi_selection():
            display_selection(cgmi_table, st.session_state.cgmi, data, existing_data_ind, label_names=get_label_names_from_fig(cgmi_fig), suffix="_cgmi")
        with cgmi_graph:
            st.plotly_chart(cgmi_fig, on_select=display_cgmi_selection, key='cgmi')
    
    # tab to display the t-SNE embeddings of each split on both datasets
    with tab3:
        tsne_fig = plotting.gen_tsne_fig(existing_data_name, new_data_name)
        tsne_graph, tsne_table = st.container(), st.container()
        def display_tsne_selection():
            display_selection(tsne_table, st.session_state.tsne, data, existing_data_ind, label_names=get_label_names_from_fig(tsne_fig), suffix="_tsne")
        with tsne_graph:
            st.plotly_chart(tsne_fig, on_select=display_tsne_selection, key='tsne')

if __name__ == "__main__":
    main()