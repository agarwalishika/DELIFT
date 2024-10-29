# DELIFT: Data Efficient Language model Instruction Fine Tuning

This repo contains the code for [DELIFT: Data Efficient Language model Instruction Fine Tuning](https://arxiv.org/abs/LINK). DELIFT is a unified fine-tuning algorithm that optimizes data subset selection across the three stages of fine-tuning:
- **Stage 1**: Instruction Tuning: enhancing a model's ability to follow general instructions
- **Stage 2**: Task-Specific Fine-Tuning: refining a model's expertise in specific domains
- **Stage 3**: Continual Fine-Tuning: integrating new information into a model while mitigating catastrophic forgetting.

## Running DELIFT
### Datasets
Dataset specifics can be defined in `huggingface_datasets.json` (used for Stage 1) and `benchmark_datasets.json` (used for Stage 2 - stage 3 uses a combination of these datasets). The following attributes can be defined for each dataset:
- "input": the column name that corresponds with the input. This is mandatory.
- "output": the column name that corresponds with the output. This is also mandatory.
-"split_name": the names of the training, validation, and testing splits.
- "instruction": [optional] the column name that corresponds with the instruction. This can be left out, and an empty instruction will be added.
- "subset": [optional] whether there is a specific subset of the data that needs to be loaded.

### Files to run

Any of the `run_....sh` files can be used to reproduce our results. Each file follows the below structure.

First, run data pre-processing with:
```
py visualization/create_embeddings.py
```

Next, load all the experimental results (this will take time):
```
py visualization/load_all_experiments.py
```

Finally, load the visualization:
```
py visualization/visualization.py
```

Note: the middle step can be skipped, as `visualization.py` also has the same code to load all the experimental results. Still, it is recommended to load all experiments before rendering the visualization.

## Citation
Please cite our paper:

(Please be patient, our work is under submission and we'd like the anonimity to remain until after the review period. Thank you!)