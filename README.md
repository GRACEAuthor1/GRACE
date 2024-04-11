# GRACE
In this work we present GRACE, a post-hoc interpretability technique specifically designed for single layer GNNs. GRACE is highly scalable and offers rapid processing capabilities for large graph applications. We show both analytically and empirically the relationship between GRACE and conventional, interpretable user-to-user and item-to-item collaborative filtering strategies




## Datasets
We run our experiment on four datasets: __Gowalla__, __Yelp2018__, __Amazon-Book__ and __Tomplay__

|     Dataset   |   # Users  | # Items| # Interactions | Density |
|---------------|------------|--------|----------------|---------|
|    Gowalla    | 29,858     | 40,981 | 1,027,370      | 0.00084 |
|   Yelp2018    | 31,668     | 38,018 | 1,561,406      | 0.00130 |
|  Amazon-Book  | 52,643     | 91,599 | 2,984,108      | 0.00062 |
|  Tomplay      | 35,028     | 33,397 | 7,510,000      | 0.00642 |

In particular __Gowalla__, __Yelp2018__, __Amazon-Book__ are three widely used dataset for the recommendation task used for example in the NGCF and LightGCN paper.



## The Recommendation Task

### Train your model. 

A LightGCN can be trained using the following command:
```bash
python RecSys/nn/run_train.py /path/to/config/file.yaml
```

The config files gather all informations about the models, its hyperparameters, the training parameters and the dataset on wich the model should be trained.

The following is a description of what can be put in the config.

Some configs are given in the folder: ```RecSys/config/```

#### 1. Experiment
- The key `name` is the name of the experiment. Metrics over training and model parameters will be saved under __RecSys/config/res/`name`/__
- The key `dataset` is to choose between the available datasets in the folder __data__.
In order to be consider, a dataset must have at least three files: __data/`dataset`/split/interactions_train.csv__, __data/`dataset`/split/interactions_val.csv__ and __data/`dataset`/split/interactions_test.csv__. Each of these files is a csv files having at least two columns representing interactions:
    - `u`: user id
    - `i`: item id
    - `t`: (optional) timestamp
- The optional key `seed` is an int that cand be useful for reproducing results.

#### 2. Model parameters
- The key `embedding_dim` allows to choose the embedding dimension.
- `num_layers` allows to choose the number of layers. 

#### 3. Training parameters
- The key `loss` can be __BPR__ as for BPR loss or __CE__ as for cross entropy loss.
- The keys `lr`, `epochs`, `batch_size` are the standard training parameters.


### Test your model

After being trained with the previous instructions, a model can simply be tested using the following command:
```bash
python RecSys/nn/run_eval.py /path/to/results/folder/
```

You can perform a ttest between the metrics of two model with the command:
```bash
python ttest.py /path/to/results/of/model1/ /path/to/results/of/model2/
```


## Interpretability

We implemented three different method for explaining GNNs:
 - SensitivityAnalysis : it measures the impact of a particular change in the input on the prediction. We used the local gradient of the model with respect to the nodes features to quantify sensitivity. 
 - GNNExplainer : it is an approach designed to explain GNN-based models. Its primary objective is to identify crucial graph structures by maximizing the mutual information between the GNN’s predictions and the distribution of subgraphs derived from the input.
 - GRACE (our work) : it leverages LightGCN uses a linear aggregation function and the (linear) dot product to compute user-item pair scores.


 The model implementations can be found in `RecSYS/interpretability/models`

 Then scripts under `RecSys/interpretability` are used to compare the different methods.
