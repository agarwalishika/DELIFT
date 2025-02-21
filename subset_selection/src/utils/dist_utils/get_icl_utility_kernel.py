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
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import math

class ModelDependentICLUtility:
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
    # def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Initialize the ModelDependentICLUtility class.

        Args:
            model: Pre-trained language model.
            tokenizer: Tokenizer corresponding to the pre-trained language model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        # self.model = model  # Convert the model to the specified device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.device = device

        self.sampling_params = SamplingParams(logprobs=15)
        self.model = LLM(model_name)

    def compute_model_prediction_probability_distances(self, prompts, ground_truths):
        """
        Compute the prediction probability distances for model outputs.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.
            token_type_ids (torch.Tensor): Tensor of token type IDs.

        Returns:
            distances (list): List of distances for each input example.
        """
        # input_ids = input_ids.to(self.model.device)
        # attention_mask = attention_mask.to(self.model.device)
        # with torch.no_grad():  # Disable gradient calculation for inference
        #     outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # logits_all = outputs.logits  # Get the logits from the model outputs
        # distances = []

        # for logits, input_id, token_type_id in zip(logits_all, input_ids, token_type_ids):
        #     # Shift logits and input_ids to align with next token prediction
        #     shifted_logits = logits[:-1, :]
        #     shifted_input_ids = input_id[1:]
        #     shifted_token_type_ids = token_type_id[1:]

        #     # Identify positions where token_type_ids == 1
        #     valid_positions = (shifted_token_type_ids == 1)

        #     if valid_positions.sum() == 0:
        #         # If no valid positions, skip this example
        #         distances.append(torch.tensor(0.0, device=logits.device))
        #         continue

        #     # Filter logits and labels by valid positions
        #     valid_logits = shifted_logits[valid_positions]
        #     valid_labels = shifted_input_ids[valid_positions]

        #     # Compute probabilities for the valid tokens
        #     probs = F.softmax(valid_logits, dim=-1)  # Apply softmax to get probabilities
        #     pred_probs = probs.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)  # Get probabilities of ground truth tokens
            
        #     # Compute Euclidean distance with length normalization
        #     num_valid_tokens = valid_positions.sum().float()
        #     distance = torch.norm(pred_probs - 1.0) / torch.sqrt(num_valid_tokens)
        #     distances.append(distance)
        # del input_ids
        # del attention_mask
        # return distances

        outputs = self.model.generate(prompts, use_tqdm=False, sampling_params=self.sampling_params)

        distances = []
        for output, gt_text in zip(outputs, ground_truths):
            token_logprobs = output.outputs[0].logprobs  # Log probs of generated tokens
            token_ids = output.outputs[0].token_ids  # Tokenized form of generated text

            # Tokenize ground truth text
            gt_tokens = self.tokenizer(gt_text, return_tensors="pt").input_ids.squeeze(0)

            # Align token lengths (truncate/pad if necessary)
            min_len = min(len(gt_tokens), len(token_ids))
            gt_tokens = gt_tokens[:min_len]
            token_logprobs = token_logprobs[:min_len]  # Extract corresponding logprobs

            pred_probs = []
            for gt, pred in zip(gt_tokens, token_logprobs):
                if gt.item() in pred.keys():
                    pred_probs.append(math.exp(pred[gt.item()].logprob))
                else:
                    pred_probs.append(0.0)
            pred_probs = torch.tensor(pred_probs)

            # Compute Euclidean distance with length normalization
            num_valid_tokens = torch.tensor(min_len, dtype=torch.float32)
            distance = torch.norm(pred_probs - 1.0) / torch.sqrt(num_valid_tokens)
            distances.append(distance.cpu())

        return distances

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
        
        n_train = len(train_prompts)
        n_valid = len(valid_prompts)
        utility_kernel = np.zeros((n_valid, n_train))

        # self.model.to('cuda')

        vllm_batch_size = 64

        instruction_no_icl="Please generate a response to the following query."
        instruction_with_icl="Use the following example as a guide to answer the query in the same format and style, providing a clear and concise response."

        # Compute distances without ICL examples
        distances_without_icl = []
        for i in tqdm(range(0, len(valid_prompts), vllm_batch_size), desc="without icl"):
            prompts_with_instruction_no_icl = [f"Instruction: {instruction_no_icl}\n\n\nQuery: {prompt}\n" for prompt in valid_prompts[i:i+vllm_batch_size]]
            distances = self.compute_model_prediction_probability_distances(
                prompts_with_instruction_no_icl, valid_responses[i:i+vllm_batch_size]
            )
            distances_without_icl.extend(distances)
        
        print('without icl distances!')

        # Compute distances with ICL examples and populate utility kernel
        # for j, icl_batches_for_example in tqdm(enumerate(with_icl_batches), total=len(with_icl_batches)):
        #     for batch, indices in zip(icl_batches_for_example, batched_original_indices):
        #         input_ids, attention_masks, token_type_ids = batch
        #         distances = self.compute_model_prediction_probability_distances(input_ids, attention_masks, token_type_ids)
        #         for dist, idx in zip(distances, indices):
        #             distance_with_icl = dist.cpu().numpy()
        #             if kernel_type == 'exponential':
        #                 utility_kernel[idx, j] = np.exp(distances_without_icl[idx] - distance_with_icl)
        #             elif kernel_type == 'euclidean':
        #                 utility_kernel[idx, j] = distances_without_icl[idx] - distance_with_icl
        #             else:
        #                 raise ValueError(f"Invalid kernel type: {kernel_type}")

        for i, data in tqdm(enumerate(zip(valid_prompts, valid_responses)), total=len(valid_prompts), desc="with icl"):
            prompt, response = data
            for j in range(0, len(train_prompts), vllm_batch_size):
                prompts_with_instruction_icl = [f"Instruction:\n{instruction_with_icl}\n\n\nExample:\n{ep}\n{er}" for ep, er in zip(train_prompts[j:j+vllm_batch_size], train_responses[j:j+vllm_batch_size])]
                prompts_with_instruction_icl = [f"{pwii}\n\n\nQuery:\n{prompt}\n" for pwii in prompts_with_instruction_icl]
                responses_icl = [response] * len(prompts_with_instruction_icl)
                distance_with_icl = self.compute_model_prediction_probability_distances(
                    prompts_with_instruction_icl, responses_icl
                )
                utility_kernel[i][j:j+vllm_batch_size] = distances_without_icl[i] - torch.stack(distance_with_icl)


        if scaling == 'min-max':
            # Scale to [0, 1] by min-max normalization
            min_val = utility_kernel.min()
            max_val = utility_kernel.max()
            utility_kernel = (utility_kernel - min_val) / (max_val - min_val)
        
        return utility_kernel

if __name__ == "__main__":
    import time
    start = time.time()
    argparser = ArgumentParser()
    argparser.add_argument("--existing_data_name", type=str, default="mix-instruct")
    argparser.add_argument("--new_data_name", type=str, default="mix-instruct")
    argparser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    argparser.add_argument('--is_data', type=str, default="True")
    args = argparser.parse_args()

    args.is_data = args.is_data == "True"
    
    model_name = args.model_name
    threshold = 0.7
    subset_percentage = 0.3
        
    # Set up data variables for general experiments
    existing_data_name = args.existing_data_name[args.existing_data_name.rfind('/')+1:]
    new_data_name = args.existing_data_name[args.existing_data_name.rfind('/')+1:]

    cache_type = "same_data_cache"
    labels = ['mix-instruct', 'Magpie-Llama-3.1-Pro-300K-Filtered']
    if "benchmark" in existing_data_name:
        cache_type = "benchmark_cache"
        labels = ['benchmark_gsm8k', 'mix-instruct', 'Magpie-Llama-3.1-Pro-300K-Filtered']
    if "alpaca" in existing_data_name:
        cache_type = "version_cache"
        labels = ['alpaca', 'chatalpaca-20k']
    fn = FolderNames(model_name, cache_type)

    # models = Models(language_model_name=model_name)


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

    queue_file = f'/shared/storage-01/users/ishikaa2/delift_crc/icl_utility_kernel_queue_{args.is_data}_{dataset_config_code}.pkl'
    utility_file = f'/shared/storage-01/users/ishikaa2/delift_crc/icl_utility_kernel_{args.is_data}_{dataset_config_code}.pkl'
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
        block_size = 100000000

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

    # dump back the queue with the first element popped
    # for running this file for all blocks, make sure that each block is run after 20-30 seconds gaps
    # with open(queue_file, 'wb') as f:
    #     pickle.dump(queue, f)
    
    print(time.time() - start)
    # calculate the block's similarity
    train_prompts = train_prompts[current_block[0]:current_block[1]]
    train_responses = train_responses[current_block[0]:current_block[1]]
    valid_prompts = valid_prompts[current_block[2]:current_block[3]]
    valid_responses = valid_responses[current_block[2]:current_block[3]]

    mod_dep = ModelDependentICLUtility(model_name, model_name)
    kernel = mod_dep.calculate_icl_utility(train_prompts, train_responses, valid_prompts, valid_responses)

    # dump the block kernel values in the correct spot
    with open(utility_file, 'rb') as f:
        full_kernel = pickle.load(f)

    full_kernel[current_block[0]:current_block[1], current_block[2]:current_block[3]] = kernel
    
    with open(utility_file, 'wb') as f:
        pickle.dump(full_kernel, f)