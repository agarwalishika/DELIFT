from tqdm import tqdm
import numpy as np
import evaluate
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rouge_metric = evaluate.load('rouge')
bleu_metric = evaluate.load('bleu')

def calculate_similarity(predictions, references, score="rouge", return_invidiual=True):
    """
    Calculates the similarity between the predictions and references

    Args:
        predictions: list of strings for the hypothesis
        references: list of strings for the reference
        score: either "rouge" or "bleu" to calculate either metric
        return_invidiual: if True, it will return the individual scores for corresponding prediction-reference pairs
    Returns:
        np array of metrics of size 1x1 if return_individual is True, else 1x|predictions|
    """
    if not return_invidiual:
        predictions = [predictions]
        references = [references]
    else:
        predictions = [[p] for p in predictions]
        references = [[r] for r in references]
    
    sim_metric = rouge_metric if score == "rouge" else bleu_metric
    metric_key = "rouge1" if score == "rouge" else "bleu"
    metrics = []
    for p, r in zip(predictions, references):
        metrics.append(sim_metric.compute(predictions=p, references=r)[metric_key])
    return np.array(metrics)
    


def perform_inference(model, tokenizer, prompts, references, batch_size=3):
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
    prompts = list(prompts)
    for j in range(len(prompts)):
        prompts[j] += "Output:"

    metrics = []
    generated_text = []
    for j in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[j:j+batch_size]
        batch_ref = references[j:j+batch_size]

        tokenized_output = tokenizer(batch_prompts, padding=True, return_tensors='pt').to(device)
        tokenized_output['input_ids'] = tokenized_output['input_ids'][:, :tokenizer.model_max_length-100]
        tokenized_output['attention_mask'] = tokenized_output['attention_mask'][:, :tokenizer.model_max_length-100]
        gen_tokens = model.generate(
            **tokenized_output,
            max_length=tokenizer.model_max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature=0.8
        )

        # decode
        for x in range(batch_size):
            prompt_length = len(tokenized_output[x])
            gen_text = tokenizer.decode(gen_tokens[x][prompt_length:]).strip()
            ## take only the first line of the generated text
            gen_text = gen_text[:gen_text.find('\n')].strip()

            # evaluate
            sim = rouge_metric.compute(predictions=[gen_text], references=[batch_ref[x].strip()])['rouge1']
            metrics.append(sim)
            generated_text.append(gen_text)
        
        del tokenized_output, gen_text
        torch.cuda.empty_cache()

    metrics = np.array(metrics)
    return metrics, generated_text