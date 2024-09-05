from data_object import DataObject, DataObjectConstants
from transformers import AutoModelForCausalLM
from folder_names import FolderNames
import plotly.graph_objs as go
from lm_knowledge import *
from models import Models
import numpy as np
import submodlib
import pickle
import torch
import sys
import os
sys.path.append('.')
from subset_selection.src.utils.dist_utils.get_similarity_kernel_torch import ModelIndependentICLUtility
from subset_selection.src.utils.dist_utils.get_icl_utility_kernel import ModelDependentICLUtility
from subset_selection.src.utils.dist_utils.select_it_baseline import SelectIT
from subset_selection.subset_random import RandomSubsetCreation
from subset_selection.inference_peft import InferencePEFT
from subset_selection.subset_fl import FLSubsetCreation 
from subset_selection.inference_icl import InferenceICL
import subset_selection.model_inference as mi

class Plotting():
    def __init__(self, data: DataObject, labels, models: Models, fn: FolderNames):
        """
        Creates Plotting instance.

        Args:
            data: instance of DataObject from data_object.py
            labels: list of data set names available in the cache/dataset_pkls
            models: instance of Models from models.py
            fn: instance of FolderNames from folder_names.py
        """
        self.labels = labels
        self.data = data
        self.models = models
        self.fn = fn

    def gen_tsne_fig(self, existing_data_name, new_data_name):
        """
        Creates a Plotly graph object figure for displaying t-SNE embeddings. This will go in the Data Point Viewer tab on the visualization.

        Args:
            existing_data_name: name of the existing dataset
            new_data_name: name of the new dataset (could be the same dataset name)
        """
        # helper method to create each scatterplot
        def create_scatter(curr_data, point_labels, name, color):
            x_data = curr_data[:, 0]
            y_data = curr_data[:, 1]
            if "train" in name:
                symbol = 'x'
            elif "valid" in name:
                symbol = 'circle'
            else:
                symbol = 'cross'
            
            return go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                marker=dict(color=color, opacity=0.95, size=8, symbol=symbol),
                text=point_labels,  # Text for hover
                hoverinfo='text',
                name=name
            )
        
        # create all the scatter plots
        scatterplots = []
        scatterplots.append(create_scatter(self.data.train_new_vis_dims, self.data.train_new_point_labels, f"training: {new_data_name}", "dodgerblue"))
        scatterplots.append(create_scatter(self.data.valid_new_vis_dims, self.data.valid_new_point_labels, f"validation: {new_data_name}", "dodgerblue"))
        scatterplots.append(create_scatter(self.data.test_new_vis_dims, self.data.test_new_point_labels, f"testing: {new_data_name}", "dodgerblue"))
        scatterplots.append(create_scatter(self.data.train_existing_vis_dims, self.data.train_existing_point_labels, f"training: {existing_data_name}", "orange"))
        scatterplots.append(create_scatter(self.data.valid_existing_vis_dims, self.data.valid_existing_point_labels, f"validation: {existing_data_name}", "orange"))
        scatterplots.append(create_scatter(self.data.test_existing_vis_dims, self.data.test_existing_point_labels, f"testing: {existing_data_name}", "orange"))

        # create the figure
        layout = go.Layout(
            title='Clickable t-SNE Embeddings',
            xaxis=dict(title='t-SNE Component 1'),
            yaxis=dict(title='t-SNE Component 2'),
            legend=dict(orientation='h', y=-0.25)
        )

        fig = go.Figure(data=scatterplots, layout=layout)
        return fig

    def gen_cg_fig(self, data: DataObject, utility_file, k_perc):
        """
        Creates a Plotly graph object figure for the Data Analysis tab which illustrates the points with conditional gain and mutual information.

        Args:
            data: instance of DataObject from data_object.py
            utility_file: the path to the file with the utility information (data_sijs and private_sijs)
            k_perc: percentage of training data chosen for the subset
        """

        with open(utility_file, 'rb') as f:
            data_sijs, private_sijs = pickle.load(f)

        n, num_privates = private_sijs.shape

        # conditional gain
        fl_cg = submodlib.functions.facilityLocationConditionalGain.FacilityLocationConditionalGainFunction(n=n, num_privates=num_privates, data_sijs=data_sijs, private_sijs=private_sijs)
        flcg_subset = fl_cg.maximize(int(k_perc * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=True)
        flcg_subset = np.array(flcg_subset)[:, 0].astype(int)

        # mutual information
        fl_mi = submodlib.functions.facilityLocationMutualInformation.FacilityLocationMutualInformationFunction(n=n, num_queries=num_privates, data_sijs=data_sijs, query_sijs=private_sijs)
        flmi_subset = fl_mi.maximize(int(k_perc * n), optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=True)
        flmi_subset = np.array(flmi_subset)[:, 0].astype(int)

        scatterplots = []
        point_labels = []

        # helper function to create a scatterplot
        def create_combined_scatter(set, set_labels, color, name, opacity=0.95, symbol='circle', legend="legend"):
            set_x = set[:, 0]
            set_y = set[:, 1]
            scatter = go.Scatter(
                x=set_x,
                y=set_y,
                mode='markers',
                marker=dict(color=color, opacity=opacity, size=8, symbol=symbol),
                text=set_labels,  # Text for hover
                hoverinfo='text',
                name=name,
            )
            return scatter
        
        # plot private set (existing data)
        scatter = create_combined_scatter(data.train_existing_vis_dims, data.train_existing_point_labels, "grey", "Existing data", opacity=0.25, symbol='x')
        scatterplots.append(scatter)

        # plot ground set (data to add)
        new_train_idx = list(set(np.arange(len(data.train_new_vis_dims))) - set(flcg_subset) - set(flmi_subset))
        scatter = create_combined_scatter(data.train_new_vis_dims[new_train_idx], data.train_new_point_labels[new_train_idx], "black", "New data (to add in)", opacity=0.5)
        scatterplots.append(scatter)
        
        # plot conditional gain subset
        unique_flcg_subset = list(set(flcg_subset) - set(flmi_subset))
        scatter = create_combined_scatter(data.train_new_vis_dims[unique_flcg_subset], data.train_new_point_labels[unique_flcg_subset], color="limegreen", name="New data that increases utility")
        scatterplots.append(scatter)

        # plot mutual information
        unique_flmi_subset = list(set(flmi_subset) - set(flcg_subset))
        scatter = create_combined_scatter(data.train_new_vis_dims[unique_flmi_subset], data.train_new_point_labels[unique_flmi_subset], color="red", name="New Data that shares knowledge")
        scatterplots.append(scatter)

        # plot intersection
        intersection = np.array(list(set(flcg_subset) & set(flmi_subset)))
        scatter = create_combined_scatter(data.train_new_vis_dims[intersection], data.train_new_point_labels[intersection], color="saddlebrown", name="New data that increases utility AND shares knowledge")
        scatterplots.append(scatter)

        # layer scatterplots together
        layout = go.Layout(
            title='What points add utility and which don\'t?',
            xaxis=dict(title='t-SNE Component 1'),
            yaxis=dict(title='t-SNE Component 2'),
            legend=dict(orientation='h', y=-0.25),
            #legend2=dict(orientation='h')#, y=-0.35)
        )

        fig = go.Figure(data=scatterplots, layout=layout)
        return fig, point_labels

    def plot_knowledge(self, label_ind, prompts, references, knowledge_file, metrics_file, threshold=0.7, prefix="BEFORE"):
        """
        Creates a Plotly graph object figure that visualizes the (given) data samples that the langauge model performs well on.

        Args:
            label_ind: index of the label for the name of the new dataset
            prompts: prompts to perform inference on
            references: references to compare model inference on
            knowledge_file: path to file where the inference results (generated text) needs to be stored
            metrics_file: path to file where the inference results (numerical ROUGE/BLEU scores for each data sample) needs to be stored
            threshold: number between 0 and 1 that indicates good/bad performance. Default is 0.7
            prefix: "BEFORE" or "AFTER" to indicate that the results are either before or after an experiment
        """
        # check to see if inference results are there - if not, perform inference and store results
        if not os.path.exists(knowledge_file):
            metrics, generated_text = perform_inference(self.models.language_model, self.models.language_tokenizer, prompts, references)

            with open(knowledge_file, 'wb+') as f:
                pickle.dump(generated_text, f)
            with open(metrics_file, 'wb+') as f:
                pickle.dump(metrics, f)
        else:
            if not os.path.exists(metrics_file):
                with open(knowledge_file, 'rb') as f:
                    generated_text = pickle.load(f)
                metrics = calculate_similarity(generated_text, references)
                with open(metrics_file, 'wb+') as f:
                    pickle.dump(metrics, f)
            else:
                with open(metrics_file, 'rb') as f:
                    metrics = pickle.load(f)
        
        # specify sets of data that the model performs well on (covered) and does not perform well on (uncovered)
        covered = self.data.valid_new_vis_dims[metrics > threshold]
        uncovered = self.data.valid_new_vis_dims[metrics <= threshold]

        scatterplots = []
        # first we plot uncovered
        labels = self.data.valid_new_point_labels[metrics <= threshold]
        scatter = go.Scatter(x=uncovered[:, 0], y=uncovered[:, 1], mode='markers', marker=dict(color='red', opacity=0.95, size=8, symbol='x'), text=labels,  hoverinfo='text', name=self.labels[label_ind] + " - below performance threshold")
        scatterplots.append(scatter)

        # second, we plot covered
        labels = self.data.valid_new_point_labels[metrics > threshold]
        scatter = go.Scatter(x=covered[:, 0], y=covered[:, 1], mode='markers', marker=dict(color="green", opacity=0.95, size=8, symbol="cross"), text=labels,  hoverinfo='text', name=self.labels[label_ind] + " - above performance threshold")
        scatterplots.append(scatter)

        layout = go.Layout(
            title=f'{prefix}: What does the LM know?',
            xaxis=dict(title='t-SNE Component 1'),
            yaxis=dict(title='t-SNE Component 2'),
            legend=dict(orientation='h', y=-0.25)
        )

        fig = go.Figure(data=scatterplots, layout=layout)
        return fig, metrics > threshold, metrics <= threshold

    def visualize_subset(self, k, utility_criteria, data: DataObject):
        """
        Creates a Plotly graph object figure that visualizes the subset chosen.

        Args:
            k: percentage of training set to use as the subset
            utility_crtiera: model dependent utility, model dependent gradients, or model independent
            data: instance of DataObject from data_object.py
        """
        if "initial" in utility_criteria.lower():
            return None, []
        
        if "SelectIT" in utility_criteria:
            subset_file = self.fn.select_it_subset_file(data.dataset_config_code)

            if not os.path.exists(subset_file):
                print("Calculating SelectIT scores...")
                selectIT = SelectIT(self.models.language_model, self.models.language_tokenizer)
                subset = selectIT.get_subset(data.train_new_prompts, data.train_new_references)

                with open(subset_file, 'wb+') as f:
                    pickle.dump(subset, f)

            with open(subset_file, 'rb') as f:
                subset = pickle.load(f) 


        # calculate model dependent icl utility
        if "Model Dependent" in utility_criteria: #if "ICL Utility" in utility_criteria:
            utility_file = self.fn.model_dep_utility_file(data.dataset_config_code) # self.fn.model_dep_icl_utility_file(data.dataset_config_code)
            if not os.path.exists(utility_file):
                print(utility_criteria, ': Calculating model dependent icl utility...')
                model_dep = ModelDependentICLUtility(self.models.language_model, self.models.language_tokenizer)
                data_sijs = model_dep.calculate_icl_utility(train_prompts=data.train_new_prompts, train_responses=data.train_new_references)
                private_sijs = model_dep.calculate_icl_utility(train_prompts=data.train_existing_prompts, train_responses=data.train_existing_references, valid_prompts=data.train_new_prompts, valid_responses=data.train_new_references,)
                with open(utility_file, 'wb+') as f:
                    pickle.dump((data_sijs, private_sijs), f)
            with open(utility_file, 'rb') as f:
                data_sijs, private_sijs = pickle.load(f)
        
        # calculate model dependent gradient utility
        # elif "Gradient" in utility_criteria:
        #     utility_file = self.fn.model_dep_grad_utility_file(data.dataset_config_code)
        #     if not os.path.exists(utility_file):
        #         print(utility_criteria, ': Calculating model dependent gradient utility...')
        #         model_dep = ModelDependentGradientUtility(self.models.language_model, self.models.language_tokenizer)
        #         data_sijs = model_dep.calculate_icl_utility(train_prompts=data.train_new_prompts, train_responses=data.train_new_references)
        #         private_sijs = model_dep.calculate_icl_utility(train_prompts=data.train_existing_prompts, train_responses=data.train_existing_references, valid_prompts=data.train_new_prompts, valid_responses=data.train_new_references,)
        #         with open(utility_file, 'wb+') as f:
        #             pickle.dump((data_sijs, private_sijs), f)
        #     with open(utility_file, 'rb') as f:
        #         data_sijs, private_sijs = pickle.load(f)

        # calculate model independent semantic similarity utility
        elif "Model Independent" in utility_criteria:
            model_ind = ModelIndependentICLUtility()

            utility_file = self.fn.model_ind_utility_file(data.dataset_config_code)
            if not os.path.exists(utility_file):
                print(utility_criteria, ': Calculating model independent utility...')
                new_tensor = mi.batch_inference(self.models.embedding_model, self.models.embedding_tokenizer, list(data.train_new_data))
                existing_tensor = mi.batch_inference(self.models.embedding_model, self.models.embedding_tokenizer, list(data.train_existing_data))

                data_sijs = model_ind.compute_pairwise_similarities(new_tensor, sparse=False, batch_size=2000, device='cuda',
                                metric='cosine', scaling='additive').numpy()
                private_sijs = model_ind.compute_pairwise_similarities(new_tensor, existing_tensor, sparse=False, batch_size=2000, device='cuda',
                                metric='cosine', scaling='additive').numpy()

                with open(utility_file, 'wb+') as f:
                    pickle.dump((data_sijs, private_sijs), f)

            with open(utility_file, 'rb') as f:
                data_sijs, private_sijs = pickle.load(f)

        # choose subset based on above utility scores
        print('Finding subset...')
        if "FL" in utility_criteria:
            fl_subset = FLSubsetCreation()
            if data.use_case == DataObjectConstants.DATA_OBJECT_BENCHMARK:
                subset = fl_subset.create_mutual_information_subset(data_sijs=data_sijs, query_sijs=private_sijs, k=k)
            else:
                subset = fl_subset.create_conditional_gain_subset(data_sijs=data_sijs, private_sijs=private_sijs, k=k)
        elif "Random" in utility_criteria:
            random_subset = RandomSubsetCreation()
            subset = random_subset.create_subset(len(data.train_new_data), k)
        elif "Full" in utility_criteria:
            random_subset = RandomSubsetCreation()
            subset = random_subset.create_subset(len(data.train_new_data), 1.0)

        # display the subset
        x_data = np.array(self.data.train_new_vis_dims)[:, 0]
        y_data = np.array(self.data.train_new_vis_dims)[:, 1]

        new_data_name = data.dataset_config_code.split('|')[-1]
        subset_idx = np.array(subset)[:, 0].astype(np.int32)

        scatterplots = []

        ## overall training set
        train_idx = list(set(np.arange(len(x_data))) - set(subset_idx))
        scatter = go.Scatter(x=x_data[train_idx], y=y_data[train_idx], mode='markers', marker=dict(color='black', opacity=0.15, size=8, symbol='x'), text=data.train_new_point_labels[train_idx],  hoverinfo='text', name=new_data_name + " - training set")
        scatterplots.append(scatter)

        ## subset
        labels = data.train_new_point_labels[subset_idx]
        scatter = go.Scatter(x=x_data[subset_idx], y=y_data[subset_idx], mode='markers', marker=dict(color='black', opacity=1.0, size=8), text=labels,  hoverinfo='text', name=new_data_name + " - subset point")
        scatterplots.append(scatter)

        layout = go.Layout(
            title='What does the subset look like?',
            xaxis=dict(title='t-SNE Component 1'),
            yaxis=dict(title='t-SNE Component 2'),
            legend=dict(orientation='h', y=-0.25)
        )
        return go.Figure(data=scatterplots, layout=layout), subset_idx

    def obtain_experiment_results(self, label_ind, data: DataObject, dataset_config_code, exp_config, prefix="AFTER", threshold=0.7):
        """
        Perform inference (and store the results - generated text and corresponding metrics) on the model given a specific experiment configuration.
        """

        # set up path to files (t)
        experiment_file = self.fn.exp_knowledge_file(dataset_config_code, exp_config)
        metrics_file = self.fn.metrics_file(dataset_config_code, exp_config)

        # if ICL is the subset quality evaluation
        if "ICL" in exp_config:
            io_file = self.fn.icl_io_file(dataset_config_code, exp_config)

            # if there are ICL example-augmented prompts and references stored, create and store them
            if not os.path.exists(io_file):
                icl = InferenceICL(self.models.embedding_model, self.models.embedding_tokenizer)
                icl_prompts, icl_references = icl.create_icl_inference_data(data.exp_train_data, data.exp_valid_prompts, data.exp_valid_references)
                with open(io_file, "wb+") as f:
                    pickle.dump([icl_prompts, icl_references], f)
            with open(io_file, "rb") as f:
                io = pickle.load(f)
            
            # if there is no inference performed on the ICL example-augmented data, perform and store the results
            if not os.path.exists(experiment_file):
                metrics, generated_text = perform_inference(self.models.language_model, self.models.language_tokenizer, io[0], io[1])
                with open(experiment_file, "wb+") as f:
                    pickle.dump(generated_text, f)
                with open(metrics_file, "wb+") as f:
                    pickle.dump(metrics, f)

            # display the results in a graph
            with open(io_file, "rb") as f:
                io = pickle.load(f)
            return self.plot_knowledge(label_ind, io[0], io[1], experiment_file, metrics_file, threshold, prefix=prefix)
        
        # if PEFT is the subset quality evaluation
        if "PEFT" in exp_config:
            peft_model_dir = self.fn.peft_ft_model(dataset_config_code, exp_config)

            # if there is no PEFT model stored on the given data (in the DataObject), fine-tune and store it
            if not os.path.exists(peft_model_dir):
                peft = InferencePEFT(self.models.model_name)
                peft.fine_tune_model(data, peft_model_dir)
                

            # if there is no inference performed on the PEFT model, perform and store the results
            if not os.path.exists(experiment_file):
                lora_model = AutoModelForCausalLM.from_pretrained(peft_model_dir)
                metrics, generated_text = perform_inference(lora_model, self.models.language_tokenizer, data.valid_new_prompts, data.valid_new_references)
                with open(experiment_file, "wb+") as f:
                    pickle.dump(generated_text, f)
                with open(metrics_file, "wb+") as f:
                    pickle.dump(metrics, f)
            
            # display the results in a graph
            return self.plot_knowledge(label_ind, data.valid_new_prompts, data.valid_new_references, experiment_file, metrics_file, threshold, prefix=prefix)