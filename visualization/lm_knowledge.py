from tqdm import tqdm
import numpy as np
import evaluate
import torch
import similarity

rouge_metric = evaluate.load('rouge')
bleu_metric = evaluate.load('bleu')
bert_metric = evaluate.load('bertscore')

def calculate_similarity(predictions, references, score="rouge", return_invidiual=True):
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
        return similarity.calculate_evaluate_metric(predictions, references, score, return_invidiual)
    elif "bge" in score:
        return similarity.calculate_bge(predictions, references, return_invidiual)
    else:
        raise ValueError(f"Invalid similarity metric: {score}")
    

def perform_inference(model, tokenizer, prompts, references, batch_size=2):
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
    max_length = min(tokenizer.model_max_length, 2048) - 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to('cuda')
    model.eval()

    all_metrics = []
    all_gen_texts = []

    # Process prompts in batches
    max_len = max(len(prompts), 200)
    for i in tqdm(range(0, max_len, batch_size)):
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
        ).to(model.device)

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

    # model.to('cpu')
    return np.array(all_metrics), all_gen_texts


def perform__inference(model, tokenizer, prompts, references, batch_size=3):
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

        tokenized_output = tokenizer(batch_prompts, padding=True, return_tensors='pt').to(model.device)
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
        ).cpu()

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