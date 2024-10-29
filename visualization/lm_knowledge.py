from tqdm import tqdm
import numpy as np
import similarity
import evaluate
import pickle
import torch

rouge_metric = evaluate.load('rouge')
bleu_metric = evaluate.load('bleu')
bert_metric = evaluate.load('bertscore')

def calculate_similarity(predictions, references, score="rouge", return_individual=True):
    """
    Calculates the similarity between the predictions and references

    Args:
        predictions: list of strings for the hypothesis
        references: list of strings for the reference
        score: one of "rouge", "bleu", "bertscore", "bge", "promedeus"
        return_invidiual: if True, it will return the individual scores for corresponding prediction-reference pairs
    Returns:
        np array of metrics of size 1x1 if return_individual is True, else 1x|predictions|
    """
    if "rouge" in score or "bleu" in score or "bertscore" in score:
        return similarity.calculate_evaluate_metric(predictions, references, score, return_individual)
    elif "bge" in score:
        return similarity.calculate_bge(predictions, references, return_individual)
    elif "prometheus" in score:
        return similarity.calculate_prometheus(predictions, references, return_individual)
    else:
        raise ValueError(f"Invalid similarity metric: {score}")
    

def perform_inference(model, tokenizer, prompts, references, batch_size=2, save_path=None):
    """
    Performs inference on prompts and computes the ROUGE between the generated text and corresponding reference

    Args:
        model: AutoModelForCausalLM instance
        tokenizer: AutoTokenizer instance with left padding enabled
        prompts: list of prompts
        references: list of references
    Return:
        metrics: np array of ROUGE-1 metrics of size 1x|prompts|
        generated_text: array of size 1x|prompts| of generated responses from the given LM
    """
    max_length = int(tokenizer.model_max_length)
    device = model.module.device if type(model) == torch.nn.DataParallel else model.device
    model.eval()

    all_metrics = []
    all_gen_texts = []

    # Process prompts in batches
    max_len = min(len(prompts), 200)
    for i in tqdm(range(0, max_len, batch_size), total=max_len):
        batch_prompts = prompts[i:i+batch_size]
        batch_references = references[i:i+batch_size]

        # Modify prompts in place
        for j, prompt in enumerate(batch_prompts):
            batch_prompts[j] = prompt + "\n### Output:\n"

        tokenized_output = tokenizer(
            list(batch_prompts), 
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(
                **tokenized_output,
                max_new_tokens=150,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                min_p=0.1,
                temperature=0.2,
            )

        # Extract only the new tokens
        new_tokens = gen_tokens[:, tokenized_output.input_ids.shape[1]:]
        
        # Decode only the new tokens
        batch_gen_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        batch_gen_texts = [gen_text.replace("\n", " ") for gen_text in batch_gen_texts]
        # Calculate metrics for the batch
        batch_metrics = [rouge_metric.compute(predictions=[gen_text], references=[ref.strip()])['rouge1'] for gen_text, ref in zip(batch_gen_texts, batch_references)]

        all_metrics.extend(batch_metrics)
        all_gen_texts.extend(batch_gen_texts)

        if save_path:
            with open(save_path, "wb+") as f:
                    pickle.dump(all_gen_texts, f)

    return np.array(all_metrics), all_gen_texts