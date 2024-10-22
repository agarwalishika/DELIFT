import numpy as np

def get_prompts_refs(data):
    """
    Split text data into prompts and references by the position of the string "Output:".

    Args:
        list of text data points
    Returns:
        list of corresponding prompts and references
    """
    def extract_prompt(x):
        x = str(x)
        s = 'Output:'
        ind = x.index(s)
        return x[:ind]

    def extract_ref(x):
        x = str(x)
        s = 'Output:'
        ind = x.index(s)
        return x[ind+len(s):]
    prompts = np.vectorize(extract_prompt)(data)
    references = np.vectorize(extract_ref)(data)
    return prompts, references

class DataObjectConstants:
    """
    Constants used in the DataObject class.
    """

    # indicates the data object is used for cases where the new and existing datasets are the same
    DATA_OBJECT_SAME_DATSET = 0

    # indicates the data object is used for cases where the new dataset is a newer version of the existing dataset
    DATA_OBJECT_NEW_VERSION = 1

    # indicates the data object is used for cases where the new dataset is a benchmark and we want to improve the performance of the existing dataset on the benchmark
    DATA_OBJECT_BENCHMARK = 2

class DataObject():
    def __init__(self, existing_data_name, existing_data_ind, new_data_name, new_data_ind, existing_data, existing_vis_dims, existing_point_labels, new_data, new_vis_dims, new_point_labels, case):
        """
        Create DataObject instance.

        Args:
            existing_data_name: name of EXISTING dataset
            existing_data_ind: index of EXISTING dataset
            new_data_name: name of NEW dataset
            new_data_ind: index of NEW dataset
            existing data: combination of prompts and references (in one array element) from the EXISTING dataset
            existing_vis_dims: t-SNE embeddings of the EXISTING data for the visualization
            existing_point_labels: point labels ('0-123') of the points in the EXISTING data for the visualization
            new data: combination of prompts and references (in one array element) from the NEW dataset
            new_vis_dims: t-SNE embeddings of the NEW data for the visualization
            new_point_labels: point labels ('0-123') of the points in the NEW data for the visualization
            case: one of DataObjectConstant values, dictates the use case required for the DataObject (see DataObjectConstants for more details)
        """
        self.existing_data_name = existing_data_name
        self.existing_data_ind = existing_data_ind
        self.new_data_name = new_data_name
        self.new_data_ind =new_data_ind
        self.use_case = case

        if case == DataObjectConstants.DATA_OBJECT_SAME_DATSET: # mix instruct
            self.init_same_dataset(existing_data[0], existing_vis_dims[0], existing_point_labels[0], new_data[0], new_vis_dims[0], new_point_labels[0])
        elif case == DataObjectConstants.DATA_OBJECT_NEW_VERSION: # ibm and gov qr
            self.init_new_version(existing_data, existing_vis_dims, existing_point_labels, new_data, new_vis_dims, new_point_labels)
        elif case == DataObjectConstants.DATA_OBJECT_BENCHMARK: # benchmark
            self.init_benchmark(existing_data, existing_vis_dims, existing_point_labels, new_data, new_vis_dims, new_point_labels)

   
    def init_same_dataset(self, existing_data, existing_vis_dims, existing_point_labels, new_data, new_vis_dims, new_point_labels):
        """
        Assume no separate training and validation set within the existing/new data sets (fits use case where we are given a dataset, and we want to choose a subset that will improve our model's performance on the dataset)
        """
        self.train_existing_data = existing_data
        self.valid_existing_data = self.train_existing_data
        self.train_existing_prompts, self.train_existing_references = get_prompts_refs(existing_data)
        self.valid_existing_prompts, self.valid_existing_references = self.train_existing_prompts, self.train_existing_references
        self.train_existing_vis_dims = existing_vis_dims
        self.valid_existing_vis_dims = self.train_existing_vis_dims
        self.train_existing_point_labels = existing_point_labels
        self.valid_existing_point_labels = self.train_existing_point_labels

        self.train_new_data = new_data
        self.valid_new_data = self.train_new_data
        self.train_new_prompts, self.train_new_references = get_prompts_refs(new_data)
        self.valid_new_prompts, self.valid_new_references = self.train_new_prompts, self.train_new_references
        self.train_new_vis_dims = new_vis_dims
        self.valid_new_vis_dims = self.train_new_vis_dims
        self.train_new_point_labels = new_point_labels
        self.valid_new_point_labels = self.train_new_point_labels
    
    def init_new_version(self, existing_data, existing_vis_dims, existing_point_labels, new_data, new_vis_dims, new_point_labels):
        """
        User gives the training and validation sets (fits use cases where the new data is either a benchmark or a new version of a dataset).
        """
        self.train_existing_data = existing_data[0]
        self.valid_existing_data = existing_data[1]
        self.train_existing_prompts, self.train_existing_references = get_prompts_refs(existing_data[0])
        self.valid_existing_prompts, self.valid_existing_references = get_prompts_refs(existing_data[1])
        self.train_existing_vis_dims = existing_vis_dims[0]
        self.valid_existing_vis_dims = existing_vis_dims[1]
        self.test_existing_vis_dims = existing_vis_dims[2]
        self.train_existing_point_labels = existing_point_labels[0]
        self.valid_existing_point_labels = existing_point_labels[1]
        self.test_existing_point_labels = existing_point_labels[2]
        
        self.train_new_data = new_data[0]
        self.valid_new_data = new_data[1]
        self.train_new_prompts, self.train_new_references = get_prompts_refs(new_data[0])
        self.valid_new_prompts, self.valid_new_references = get_prompts_refs(new_data[1])
        self.train_new_vis_dims = new_vis_dims[0]
        self.valid_new_vis_dims = new_vis_dims[1]
        self.test_new_vis_dims = new_vis_dims[2]
        self.train_new_point_labels = new_point_labels[0]
        self.valid_new_point_labels = new_point_labels[1]
        self.test_new_point_labels = new_point_labels[2]
    
    def init_benchmark(self, existing_data, existing_vis_dims, existing_point_labels, new_data, new_vis_dims, new_point_labels):
        """
        User gives the training and validation sets (fits use cases where the new data is either a benchmark or a new version of a dataset).
        """
        self.train_existing_data = existing_data[0]
        self.valid_existing_data = new_data[0]
        self.train_existing_prompts, self.train_existing_references = get_prompts_refs(existing_data[0])
        self.valid_existing_prompts, self.valid_existing_references = get_prompts_refs(new_data[0])
        self.train_existing_vis_dims = existing_vis_dims[0]
        self.valid_existing_vis_dims = new_vis_dims[0]
        self.test_existing_vis_dims = existing_vis_dims[2]
        self.train_existing_point_labels = existing_point_labels[0]
        self.valid_existing_point_labels = new_point_labels[0]
        self.test_existing_point_labels = existing_point_labels[2]
        

        self.train_new_data = existing_data[1]
        self.valid_new_data = new_data[1]
        self.train_new_prompts, self.train_new_references = get_prompts_refs(existing_data[1])
        self.valid_new_prompts, self.valid_new_references = get_prompts_refs(new_data[1])
        self.train_new_vis_dims = existing_vis_dims[1]
        self.valid_new_vis_dims = new_vis_dims[1]
        self.test_new_vis_dims = new_vis_dims[2]
        self.train_new_point_labels = existing_point_labels[1]
        self.valid_new_point_labels = new_point_labels[1]
        self.test_new_point_labels = new_point_labels[2]

    def create_train_subset(self, idx):
        """
        Used to create a subset from the "new" data.

        Args:
            idx: np array of indices that are in the subset.
        """
        self.train_new_data_sub = np.concatenate((self.train_new_data[idx], self.train_existing_data))
        self.train_new_prompts_sub = np.concatenate((self.train_new_prompts[idx], self.train_existing_prompts))
        self.train_new_references_sub = np.concatenate((self.train_new_references[idx], self.train_existing_references))
        self.train_new_vis_dims_sub = np.concatenate((self.train_new_vis_dims[idx], self.train_existing_vis_dims))
        self.train_new_point_labels_sub = np.concatenate((self.train_new_point_labels[idx], self.train_existing_point_labels))

    def set_icl_peft_sets(self, train_data, train_prompts, train_references, valid_prompts, valid_references):
        """
        Different stages of the subset selection experiment requires different sets of data for ICL and PEFT. Instead of passing these values into functions as parameters, we set them within the DataObject instance itself to make code more readable.

        Args:
            train_data: from the training data, combination of prompt and references
            train_prompts: prompts from the training data
            train_references: references from the training data
            valid_prompts: prompts from the validation data
            valid_references: references from the validation data
        """
        def merge(input_set):
            if type(input_set) is tuple:
                return np.concatenate(input_set)
            return input_set
        self.exp_train_data = merge(train_data)
        self.exp_train_prompts = merge(train_prompts)
        self.exp_train_references = merge(train_references)
        self.exp_valid_prompts = merge(valid_prompts)
        self.exp_valid_references = merge(valid_references)
    
    def set_dataset_config_code(self, dataset_config_code):
        self.dataset_config_code = dataset_config_code
