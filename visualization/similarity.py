import evaluate
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

rouge_metric = evaluate.load('rouge')
bleu_metric = evaluate.load('bleu')
bert_metric = evaluate.load('bertscore')

def calculate_evaluate_metric(predictions, references, score="rouge", return_invidiual=True):
    """
    Calculates the similarity (rouge, bleu, or bertscore) between the predictions and references

    Args:
        predictions: list of strings for the hypothesis
        references: list of strings for the reference
        score: one of "rouge", "bleu", "bertscore", "bge", "promedeus"
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
    

    if score == "rouge":
        sim_metric = rouge_metric
        metric_key = "rouge1"
    elif score == "bleu":
        sim_metric = bleu_metric
        metric_key = "bleu"
    else:
        sim_metric = bert_metric
        metric_key = "f1"
    
    # sim_metric = rouge_metric if score == "rouge" elif score == "bleu" bleu_metric else bert_metric
    # metric_key = "rouge1" if score == "rouge" else "bleu"
    metrics = []
    for p, r in zip(predictions, references):
        if score == "bertscore":
            metrics.append(np.array(sim_metric.compute(predictions=p, references=r, lang="en")[metric_key]).mean())
        else:
            metrics.append(sim_metric.compute(predictions=p, references=r)[metric_key])
    return np.array(metrics)



def calculate_bge(predictions, references, return_individual=True):
    embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
    embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to('cpu')
    embedding_model.eval()

    if embedding_tokenizer.pad_token is None:
        embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        embedding_model.resize_token_embeddings(len(self.embedding_tokenizer))
        
    embedding_model = embedding_model.to('cuda')

    def find_embedding(prompt):
        encoded_input = embedding_tokenizer(prompt, padding=True, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=1, dim=1).squeeze()
        return sentence_embeddings
    
    metrics = []

    for pred, ref in zip(predictions, references):
        pred_emb = find_embedding(pred)
        ref_emb = find_embedding(ref)
        metrics.append(ref_emb.dot(pred_emb).item())

    embedding_model.to('cpu')
    del embedding_model
    del embedding_tokenizer
    return np.array(metrics)