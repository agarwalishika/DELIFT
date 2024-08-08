import torch

def batch_inference(model, tokenizer, input, device):
    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt').to(device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
            
    bs = 1
    output = []
    for i in range(0, len(encoded_input['input_ids']), bs):
        output.extend(model(input_ids=input_ids[i:i+bs], attention_mask=attention_mask[i:i+bs]).pooler_output.detach())

    output = torch.stack(output)
    return output

def batch_inference_text(model, tokenizer, input, device):
    bs = 1
    output = []
    for i in range(0, len(input), bs):
        prompt = list(map(str, list(input[i:i+bs])))
        tokenized_output = tokenizer(prompt, padding=True, return_tensors='pt').to(device)
        gen_tokens = model.generate(
            **tokenized_output,
            do_sample=True,
            max_new_tokens=100,
        )
        output.extend(tokenizer.batch_decode(gen_tokens))
    return output