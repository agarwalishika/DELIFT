import os
import re

def parse_file_name(dataset_name, exp_config):
    if "before" in exp_config:
        k = re.findall(".*-.*-(.*-.*)", exp_config)[0]
        return f"before_exp-{k}_{dataset_name}"
    return f"{exp_config}_{dataset_name}"

class FolderNames:
    # query rewriting datasets
    qr_dataset_folder = 'query_rewriting_data'
    qr_gov_data_file = os.path.join(qr_dataset_folder, "gov_data.json")
    qr_ibm_ft_data_file = os.path.join(qr_dataset_folder, "ibm_ft_data.json")

    def __init__(self, model_name, cache_name="version_cache"):
        model_name = model_name[model_name.rfind('/')+1:]
        
        # main cache folder
        self.base_folder = os.path.join("visualization", cache_name)
        if not os.path.exists(self.base_folder): os.mkdir(self.base_folder)

        self.main_folder = os.path.join(self.base_folder, model_name)
        if not os.path.exists(self.main_folder): os.mkdir(self.main_folder)

        # store the embeddings of each of the data points in pkls files
        self.dataset_pkl_folder = os.path.join(self.base_folder, "dataset_pkls")
        if not os.path.exists(self.dataset_pkl_folder): os.mkdir(self.dataset_pkl_folder)

        # store the metadata of the visualization data
        self.visualization_cache_file = os.path.join(self.dataset_pkl_folder, 'all_data.pkl')

        # dataset configuration (first private sets, then the ground sets)
        self.dataset_config_file_code = lambda existing_data_name, new_data_name: f"{existing_data_name}|{new_data_name}"

        # store the knowledge before the experiments (store the generated texts)
        # self.before_exp_knowledge_file = lambda dataset_name, exp_config: os.path.join(self.main_folder, "before_exp", f"{exp_config}_{dataset_name}.pkl")
        # if not os.path.exists(os.path.join(self.main_folder, "before_exp")): os.mkdir(os.path.join(self.main_folder, "before_exp"))

        # subset creation: utility files
        self.subset_folder = os.path.join(self.main_folder, "utility")
        self.select_it_subset_file = lambda dataset_name: os.path.join(self.subset_folder, f"select_it_subset_{dataset_name}.pkl")
        self.model_dep_utility_file = lambda dataset_name: os.path.join(self.subset_folder, f"model_dep_utility_{dataset_name}.pkl")
        self.model_ind_utility_file = lambda dataset_name: os.path.join(self.subset_folder, f"model_ind_utility_{dataset_name}.pkl")
        if not os.path.exists(self.subset_folder): os.mkdir(self.subset_folder)

        # store the knowledge after the experiments
        self.exp_knowledge_file = lambda dataset_name, exp_config, prefix="": os.path.join(self.main_folder, "generated_text", f"{prefix}{parse_file_name(dataset_name, exp_config)}.pkl")
        if not os.path.exists(os.path.join(self.main_folder, "generated_text")): os.mkdir(os.path.join(self.main_folder, "generated_text"))

        # store individual similarity metrics
        self.metrics_file = lambda dataset_name, exp_config: os.path.join(self.main_folder, "metrics", f"{parse_file_name(dataset_name, exp_config)}.pkl")
        if not os.path.exists(os.path.join(self.main_folder, "metrics")): os.mkdir(os.path.join(self.main_folder, "metrics"))

        # icl specific
        self.icl_io_file = lambda dataset_name, exp_config, prefix="": os.path.join(self.main_folder, "icl_io", f"{prefix}{parse_file_name(dataset_name, exp_config)}.pkl")
        if not os.path.exists(os.path.join(self.main_folder, "icl_io")): os.mkdir(os.path.join(self.main_folder, "icl_io"))

        # peft specific
        self.peft_ft_model = lambda dataset_name, exp_config: os.path.join(self.main_folder, "peft_ft_models", parse_file_name(dataset_name, exp_config))
        if not os.path.exists(os.path.join(self.main_folder, "peft_ft_models")): os.mkdir(os.path.join(self.main_folder, "peft_ft_models"))