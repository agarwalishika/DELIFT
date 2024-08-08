# Loading Experiments

1. Configure datasets
	- To use Hugging Face datasets, all dataset reading configurations can be put in `visualization/dataset.json`.
	- If you don't want to use Hugging Face datasets, custom data manipulators should be written in `visualization/create_embeddings.py`.

2. Run `py visualization/create_embeddings.py`.
3. Run `streamlit run visualization/visualization.py`
	- If you want to load all experiments beforehand, run `py visualization/load_all_experiments.py` first. Then, start up the streamlit application, and all experiments will be cached for quick viewing.

# Motivation

Language models (LMs) are usually trained/fine-tuned with large amounts of data (in the terabytes). 

Not all of this data will be useful, or informative, to the LM as data points can be repetitive or uninformative. The goal of subset selection is, given a large pool of data, find the minimum subset of data that is as rich in information as the large data pool.

We want not only an easy way to run subset selection experiments, but also to visualize the effects of various experiments on different datasets.

# Visualizing what a LM "knows"

We define “knowledge” as points that a language model can generate text for that is semantically similar to a ground truth reference. In other words, we perform inference, compute semantic similarity between the predicted generation and the ground truth reference, and compare the similarity score with a threshold; if it’s above the threshold, we deem that as LM knowledge, else a LM limitation. This is done on the *validation set*. On the "Subset Selection" tab, the top two graphs on the left and right visualize what the LM knows before and after the experiment. The experiment configurations can be set in between the two graphs.

# Subset Selection
Our aim is to find a subset of data – from the LM limitation set – that is rich with information. 

To do this, we can use **submodular functions**. These functions help pick subsets of data that have **positive utility** (points are chosen one at a time). They also exhibit the diminishing gains property where points that are added in the subset early on have more utility than points added later. 

**Facility Location** is one such submodular function - similar to clustering - that chooses a representative set of points. As an intuitive example, suppose we have a network of nodes that all require supplies. Facility Location helps to optimally locate where to place these supply nodes. 

## Problem Formulation
Given is a training set $D_T$ and a validation set $D_V$. These sets are labeled, which means each point in these sets has an input and a reference output. Our goal is to find points in $D_T$ that represents $D_V$. 

The key to Facility Location, and submodular functions in general, is defining utility. How do we do that?


## Calculating Utility
We have devised 2 ways to calculating utility of a data point: (1) a model-independent approach, and (2) a model dependent approach.

### Model Independent
We encode data points in both $D_T$ and $D_V$ using the [BAAI general embedding model](https://huggingface.co/BAAI/bge-large-en-v1.5). For each point embedding from $D_T$, we compute the Euclidean distance to every point embedding in $D_V$ as the utility. Here, good utility is a small distance to the points in $D_V$.

### Model Dependent
A model dependent approach calculates utility based on model feedback. Specifically, we capture pairwise-interactions between points in $D_T$ and $D_V$. The question we want to answer is: does the performance on a data point improve when given an in-context example? For each pair of points $t,v$ in $D_T$ and $D_V$, we provide $t$ as an in-context example (with both input and output) and $v$'s input as a prompt to the LM.

We compute the distance between the token distributions between the predicted output and the ground truth output. We perform the same computation, except without the in-context example. The utility is calculated as the difference between these two distances, therefore, a good utility value indicates large, positive difference when given the in-context example.

## Subset Creation

Now that utility is defined on the training set, we can use Facility Location to create subsets of the large pool of data. This brings two settings called "Model Independent + Facility Location" and "Model Dependent + Facility Location". As a baseline, we can also generate a random subset to compare subset creation methods - this is simply called "Random".


## Performance of Subset
At this point, we have a subset $S_T \subset D_T$ and the validation set $D_V$. Again, we have two ways of determining this: in-context learning (ICL) and fine-tuning (PEFT).

### ICL
For each point in $D_V$, we pick the $k$ closest points in the embedding space (embeddings created with BAAI again), and add them as in-context examples. Then, we calculate the Euclidean distance between the embeddings of the predicted generation and ground truth text. We use the [paraphrase-MiniLM-L6-v2 SentenceTransformer model](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) to generate these embeddings as this model can be used for determining semantic similarity.

### PEFT
We can use LoRa to fine-tune our LM on the subset $S_T$. With the fine-tuned LM, we can again use the Euclidean distances between embeddings. 
