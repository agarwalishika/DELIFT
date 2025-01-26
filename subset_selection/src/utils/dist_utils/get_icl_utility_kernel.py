import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import sys
sys.path.append('/home/ishikaa2/delift/visualization/')
from folder_names import FolderNames
from models import Models
from data_object import DataObject, DataObjectConstants
import pickle
import os
from argparse import ArgumentParser

class ModelDependentICLUtility:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Initialize the ModelDependentICLUtility class.

        Args:
            model: Pre-trained language model.
            tokenizer: Tokenizer corresponding to the pre-trained language model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.model = model  # Convert the model to the specified device
        self.tokenizer = tokenizer
        self.device = device

    def compute_model_prediction_probability_distances(self, input_ids, attention_mask, token_type_ids):
        """
        Compute the prediction probability distances for model outputs.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.
            token_type_ids (torch.Tensor): Tensor of token type IDs.

        Returns:
            distances (list): List of distances for each input example.
        """
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits_all = outputs.logits  # Get the logits from the model outputs
        distances = []

        for logits, input_id, token_type_id in zip(logits_all, input_ids, token_type_ids):
            # Shift logits and input_ids to align with next token prediction
            shifted_logits = logits[:-1, :]
            shifted_input_ids = input_id[1:]
            shifted_token_type_ids = token_type_id[1:]

            # Identify positions where token_type_ids == 1
            valid_positions = (shifted_token_type_ids == 1)

            if valid_positions.sum() == 0:
                # If no valid positions, skip this example
                distances.append(torch.tensor(0.0, device=logits.device))
                continue

            # Filter logits and labels by valid positions
            valid_logits = shifted_logits[valid_positions]
            valid_labels = shifted_input_ids[valid_positions]

            # Compute probabilities for the valid tokens
            probs = F.softmax(valid_logits, dim=-1)  # Apply softmax to get probabilities
            pred_probs = probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)  # Get probabilities of ground truth tokens
            
            # Compute Euclidean distance with length normalization
            num_valid_tokens = valid_positions.sum().float()
            distance = torch.norm(pred_probs - 1.0) / torch.sqrt(num_valid_tokens)
            distances.append(distance)

        del input_ids
        del attention_mask
        return distances

    def prepare_batch_inputs(self, prompts, responses, example_prompts, example_responses, 
                            instruction_no_icl="Please generate a response to the following query.", 
                            instruction_with_icl="Use the following example as a guide to answer the query in the same format and style, providing a clear and concise response.",
                            max_length=2048, batch_size=32):
        """
        Prepare batches for inference with and without in-context learning examples.

        Args:
            prompts (list): List of target prompts.
            responses (list): List of target responses.
            example_prompts (list): List of in-context example prompts.
            example_responses (list): List of in-context example responses.
            instruction_no_icl (str): Instruction to be prepended to each prompt without ICL examples.
            instruction_with_icl (str): Instruction to be prepended to each prompt with ICL examples.
            max_length (int): Maximum sequence length for padding/truncation.
            batch_size (int): Number of samples per batch.

        Returns:
            without_icl_batches (list of tuples): Batches of tensors for inference without ICL examples.
            with_icl_batches (list of lists of tuples): Batches of tensors for inference with each possible ICL example.
            batched_original_indices (list of lists): List of original indices for each batch.
        """
        
        def tokenize_and_pad(prompts, responses, max_length):
            """
            Tokenize and pad the input sequences.

            Args:
                prompts (list): List of target prompts.
                responses (list): List of target responses.
                max_length (int): Maximum sequence length for padding/truncation.

            Returns:
                input_ids (torch.Tensor): Tensor of tokenized input IDs.
                attention_masks (torch.Tensor): Tensor of attention masks.
                token_type_ids (torch.Tensor): Tensor of token type IDs.
            """
            input_ids = []
            attention_masks = []
            token_type_ids = []

            # Determine the batch-specific maximum length
            batch_lengths = [len(self.tokenizer(prompt, add_special_tokens=False)['input_ids']) +
                            len(self.tokenizer(response, add_special_tokens=False)['input_ids'])
                            for prompt, response in zip(prompts, responses)]
            batch_max_length = min(max_length, max(batch_lengths))

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            for prompt, response in zip(prompts, responses):
                # Tokenize prompt and response separately
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
                response_tokens = self.tokenizer(response, add_special_tokens=False)

                # Concatenate prompt and response tokens with eos_token_id
                combined_input_ids = prompt_tokens['input_ids'] + [self.tokenizer.eos_token_id] + response_tokens['input_ids'] + [self.tokenizer.eos_token_id]
                combined_attention_mask = prompt_tokens['attention_mask'] + [1] + response_tokens['attention_mask'] + [1]

                # Generate token_type_ids: 0 for prompt, 1 for response
                combined_token_type_ids = [0] * len(prompt_tokens['input_ids']) + [0] + [1] * len(response_tokens['input_ids']) + [1]

                # Pad to the batch-specific maximum length
                padding_length = batch_max_length - len(combined_input_ids)
                if padding_length > 0:
                    combined_input_ids += [self.tokenizer.pad_token_id] * padding_length
                    combined_attention_mask += [0] * padding_length
                    combined_token_type_ids += [0] * padding_length  # Padding tokens are marked with 0

                # Truncate if the sequence is longer than batch_max_length
                combined_input_ids = combined_input_ids[:batch_max_length]
                combined_attention_mask = combined_attention_mask[:batch_max_length]
                combined_token_type_ids = combined_token_type_ids[:batch_max_length]

                # Append to the batch lists
                input_ids.append(combined_input_ids)
                attention_masks.append(combined_attention_mask)
                token_type_ids.append(combined_token_type_ids)

            return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(token_type_ids)

        # Prepend instruction to each prompt without ICL examples
        prompts_with_instruction_no_icl = [f"Instruction: {instruction_no_icl}\n\n\nQuery: {prompt}\n" for prompt in prompts]

        # Calculate lengths of each combined sequence without ICL
        lengths = [len(self.tokenizer(prompt, add_special_tokens=False)['input_ids']) + len(self.tokenizer(response, add_special_tokens=False)['input_ids']) for prompt, response in zip(prompts_with_instruction_no_icl, responses)]

        # Sort indices by lengths
        sorted_indices = np.argsort(lengths)

        # Calculate quantiles
        num_batches = (len(prompts) + batch_size - 1) // batch_size  # Ensures rounding up
        quantiles = np.array_split(sorted_indices, num_batches)

        # Prepare batches for inference without ICL examples
        without_icl_batches = []
        batched_original_indices = []

        for batch_indices in quantiles:
            batch_prompts = [prompts_with_instruction_no_icl[i] for i in batch_indices]
            batch_responses = [responses[i] for i in batch_indices]
            batch_original_indices = list(batch_indices)  # Keep track of original indices

            input_ids, attention_masks, token_type_ids = tokenize_and_pad(batch_prompts, batch_responses, max_length)

            without_icl_batches.append((input_ids, attention_masks, token_type_ids))
            batched_original_indices.append(batch_original_indices)

        # Prepare batches for inference with each possible ICL example
        with_icl_batches = []

        for example_prompt, example_response in zip(example_prompts, example_responses):
            example = f"Instruction:\n{instruction_with_icl}\n\n\nExample:\n{example_prompt}\n{example_response}"
            with_icl_batches_for_example = []
            for batch_indices in quantiles:
                batch_prompts = [f"{example}\n\n\nQuery:\n{prompts[i]}\n" for i in batch_indices]
                batch_responses = [responses[i] for i in batch_indices]

                input_ids, attention_masks, token_type_ids = tokenize_and_pad(batch_prompts, batch_responses, max_length)

                with_icl_batches_for_example.append((input_ids, attention_masks, token_type_ids))

            with_icl_batches.append(with_icl_batches_for_example)

        return without_icl_batches, with_icl_batches, batched_original_indices

    def calculate_icl_utility(self, 
                              train_prompts, 
                              train_responses, 
                              valid_prompts=None, 
                              valid_responses=None,
                              kernel_type='euclidean',
                              scaling='min-max'):
        """
        Calculate the in-context learning (ICL) utility for a given set of prompts and responses.

        Args:
            train_prompts (list): List of training prompts.
            train_responses (list): List of training responses.
            valid_prompts (list, optional): List of validation prompts. Defaults to None.
            valid_responses (list, optional): List of validation responses. Defaults to None.
            kernel_type (str): Type of kernel to use for calculating utility ('euclidean' or 'exponential').
            scaling (str): Method to scale the utility values ('min-max').

        Returns:
            utility_kernel (np.ndarray): Utility kernel matrix.
            u_{ij}  means how much the j-th example in the training set is useful for the i-th example in the validation set.
        """
        if valid_prompts is None or valid_responses is None:
            valid_prompts = train_prompts
            valid_responses = train_responses

        # Prepare batch inputs
        without_icl_batches, with_icl_batches, batched_original_indices = self.prepare_batch_inputs(
            valid_prompts, valid_responses, train_prompts, train_responses
        )
        print('batched!')
        
        n_train = len(train_prompts)
        n_valid = len(valid_prompts)
        utility_kernel = np.zeros((n_valid, n_train))

        # self.model.to('cuda')

        # Compute distances without ICL examples
        distances_without_icl = [0] * n_valid
        for batch, indices in zip(without_icl_batches, batched_original_indices):
            input_ids, attention_masks, token_type_ids = batch
            distances = self.compute_model_prediction_probability_distances(input_ids, attention_masks, token_type_ids)
            for dist, idx in zip(distances, indices):
                distances_without_icl[idx] = dist.cpu().numpy()
        
        print('without icl distances!')

        # Compute distances with ICL examples and populate utility kernel
        for j, icl_batches_for_example in tqdm(enumerate(with_icl_batches), total=len(with_icl_batches)):
            for batch, indices in zip(icl_batches_for_example, batched_original_indices):
                input_ids, attention_masks, token_type_ids = batch
                distances = self.compute_model_prediction_probability_distances(input_ids, attention_masks, token_type_ids)
                for dist, idx in zip(distances, indices):
                    distance_with_icl = dist.cpu().numpy()
                    if kernel_type == 'exponential':
                        utility_kernel[idx, j] = np.exp(distances_without_icl[idx] - distance_with_icl)
                    elif kernel_type == 'euclidean':
                        utility_kernel[idx, j] = distances_without_icl[idx] - distance_with_icl
                    else:
                        raise ValueError(f"Invalid kernel type: {kernel_type}")

        if scaling == 'min-max':
            # Scale to [0, 1] by min-max normalization
            min_val = utility_kernel.min()
            max_val = utility_kernel.max()
            utility_kernel = (utility_kernel - min_val) / (max_val - min_val)
        
        # self.model.to('cpu')
        
        return utility_kernel

if __name__ == "__main__":
    import time
    start = time.time()
    argparser = ArgumentParser()
    argparser.add_argument('--is_data', type=str, default="True")
    args = argparser.parse_args()

    args.is_data = args.is_data == "True"
    
    model_name = "microsoft/Phi-3-mini-128k-instruct" #"Qwen/Qwen2-72B-Instruct"
    threshold = 0.7
    subset_percentage = 0.3
        
    # Set up data variables for general experiments
    fn = FolderNames(model_name, 'same_data_cache')
    models = Models(language_model_name=model_name)
    labels = [label.split('.')[0] for label in os.listdir(fn.dataset_pkl_folder) if 'all_data' not in label]

    existing_data_name = 'mix-instruct'
    new_data_name = 'mix-instruct'

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
    dataset_config_code = fn.dataset_config_file_code(existing_data_name, new_data_name)
    data.set_dataset_config_code(dataset_config_code)

    queue_file = f'icl_utility_kernel_queue_{args.is_data}_{dataset_config_code}.pkl'
    utility_file = f'icl_utility_kernel_{args.is_data}_{dataset_config_code}.pkl'
    if args.is_data:
        train_prompts=data.train_new_prompts
        train_responses=data.train_new_references
        valid_prompts=data.train_new_prompts
        valid_responses=data.train_new_references
    else:
        train_prompts=data.train_existing_prompts
        train_responses=data.train_existing_references
        valid_prompts=data.train_new_prompts
        valid_responses=data.train_new_references
    
    if not os.path.exists(queue_file):
        # create the queue of blocks
        n, m = len(train_prompts), len(valid_prompts)
        block_size = 10000

        queue = []
        for i in range(0, n, block_size):
            for j in range(0, m, block_size):
                queue.append([
                    i, i+block_size,
                    j, j+block_size
                ])
        with open(queue_file, 'wb+') as f:
            pickle.dump(queue, f)
        del queue

        # create the empty similarity kernel
        full_kernel = np.zeros((n, m))
        with open(utility_file, 'wb+') as f:
            pickle.dump(full_kernel, f)

    # load the queue
    with open(queue_file, 'rb') as f:
        queue = pickle.load(f)

    # check if the queue is empty -> if it is, then we are done
    if len(queue) == 0:
        print("Done with computing kernel for: ", args.is_data, dataset_config_code)
        sys.exit(14)

    # pop the first element from the queue
    current_block = queue.pop(0)
    with open("evaluation.txt", "a+") as f:
        f.write(f"{args.is_data}, {current_block}\n")

    # dump back the queue with the first element popped
    # for running this file for all blocks, make sure that each block is run after 20-30 seconds gaps
    with open(queue_file, 'wb') as f:
        pickle.dump(queue, f)
    
    print(time.time() - start)
    # calculate the block's similarity
    train_prompts = train_prompts[current_block[0]:current_block[1]]
    train_responses = train_responses[current_block[0]:current_block[1]]
    valid_prompts = valid_prompts[current_block[2]:current_block[3]]
    valid_responses = valid_responses[current_block[2]:current_block[3]]
    
    mod_dep = ModelDependentICLUtility(models.language_model, models.language_tokenizer)
    kernel = mod_dep.calculate_icl_utility(train_prompts, train_responses, valid_prompts, valid_responses)

    # dump the block kernel values in the correct spot
    with open(utility_file, 'rb') as f:
        full_kernel = pickle.load(f)

    full_kernel[current_block[0]:current_block[1], current_block[2]:current_block[3]] = kernel
    
    with open(utility_file, 'wb') as f:
        pickle.dump(full_kernel, f)