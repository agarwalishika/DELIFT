import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch

class SuperfilteringUtility:
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

    def compute_length_normalized_log_probabilities(self, input_ids, attention_mask, token_type_ids):
        """
        Compute the length-normalized cumulative log probability of the response tokens.

        Args:
            input_ids (torch.Tensor): Tensor of input IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.
            token_type_ids (torch.Tensor): Tensor of token type IDs.

        Returns:
            log_probs (list): List of length-normalized cumulative log probabilities for each input example.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits_all = outputs.logits
        log_probs = []

        for logits, input_id, token_type_id in zip(logits_all, input_ids, token_type_ids):
            # Shift logits and input_ids to align with next token prediction
            shifted_logits = logits[:-1, :]
            shifted_input_ids = input_id[1:]
            shifted_token_type_ids = token_type_id[1:]

            # Identify positions where token_type_ids == 1 (response tokens)
            valid_positions = (shifted_token_type_ids == 1)

            if valid_positions.sum() == 0:
                # If no valid positions, skip this example
                log_probs.append(torch.tensor(0.0, device=self.device))
                continue

            # Number of response tokens
            num_response_tokens = valid_positions.sum().item()

            # Filter logits and labels by valid positions
            valid_logits = shifted_logits[valid_positions]
            valid_labels = shifted_input_ids[valid_positions]

            # Compute log probabilities for the valid tokens
            log_probs_tokens = F.log_softmax(valid_logits, dim=-1)
            token_log_probs = log_probs_tokens.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)

            # Sum log probabilities to get the cumulative log probability
            total_log_prob = token_log_probs.sum()

            # Length normalization: divide by the number of response tokens
            length_normalized_log_prob = total_log_prob / num_response_tokens

            log_probs.append(length_normalized_log_prob)

        del input_ids
        del attention_mask
        return log_probs

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
