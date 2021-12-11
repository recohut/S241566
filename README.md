# Debiased Explainable Pairwise Ranking from Implicit Feedback
Pytorch implementation of the paper "Debiased Explainable Pairwise Ranking from Implicit Feedback".<br>
Accepted at RecSys '21.

## Authors
Khalil Damak, University of Louisville.<br>
Sami Khenissi, University of Louisville.<br>
Olfa Nasraoui, University of Louisville.<br>

## Abstract
Recent work in recommender systems has emphasized the importance of fairness, with a particular interest in bias and transparency, in addition to predictive accuracy. In this paper, we focus on the state of the art pairwise ranking model, Bayesian Personalized Ranking (BPR), which has previously been found to  outperform pointwise models in predictive accuracy while also being able to handle implicit feedback. Specifically, we address two limitations of BPR: (1) BPR is a black box model that does not explain its outputs, thus limiting the user's trust in the recommendations, and the analyst's ability to scrutinize a model's outputs; and (2) BPR is vulnerable to exposure bias due to the data being Missing Not At Random (MNAR). This exposure bias usually translates into an unfairness against the least popular items because they risk being under-exposed by the recommender system.
In this work, we first propose a novel explainable loss function and a corresponding Matrix Factorization-based model called Explainable Bayesian Personalized Ranking (EBPR) that generates recommendations along with item-based explanations. Then, we theoretically quantify  additional exposure bias resulting from the explainability, and use it as a basis to propose an unbiased estimator for the ideal EBPR loss. Finally, we perform an empirical study on three real-world datasets that demonstrate the advantages of our proposed models.

## Environment settings
We use Pytorch 1.7.1.

## Description
This repository includes the code necessary to:
* <b>Train BPR [1], UBPR [2], EBPR, pUEBPR and UEBPR:</b>

```
python -m Code.train_EBPR [-h] [--model MODEL] [--dataset DATASET]
                          [--num_epoch NUM_EPOCH] [--batch_size BATCH_SIZE]
                          [--num_latent NUM_LATENT]
                          [--l2_regularization L2_REGULARIZATION]
                          [--weight_decay WEIGHT_DECAY]
                          [--neighborhood NEIGHBORHOOD] [--top_k TOP_K] [--lr LR]
                          [--optimizer OPTIMIZER] [--sgd_momentum SGD_MOMENTUM]
                          [--rmsprop_alpha RMSPROP_ALPHA]
                          [--rmsprop_momentum RMSPROP_MOMENTUM]
                          [--loo_eval LOO_EVAL] [--test_rate TEST_RATE]
                          [--use_cuda USE_CUDA] [--device_id DEVICE_ID]
                          [--save_models SAVE_MODELS] [--int_per_item INT_PER_ITEM]
```

The code is set up to train EBPR on the Movielens 100K dataset. You can change the model using the "model" argument. Also, you can change the "dataset" argument to choose between the "Movielens 100K", "Movielens 1M", "Yahoo! R3" or "Last.FM 2K" datasets. The model will train and output the NDCG@K, HR@K, MEP@K, WMEP@K, Avg_Pop@K, EFD@K, and Div@K results on the test set for every epoch using the Leave-One-Out (LOO) evaluation procedure. You can choose the standard random train/test split by changing the parameter "loo_eval" in the "config" dictionary. The list of arguments is presented below:

```
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR',
                        'UEBPR'.
  --dataset DATASET     'ml-100k' for Movielens 100K. 'ml-1m' for the
                        Movielens 1M dataset. 'lastfm-2k' for the Last.FM 2K
                        dataset. 'yahoo-r3' for the Yahoo! R3 dataset.
  --num_epoch NUM_EPOCH
                        Number of training epochs.
  --batch_size BATCH_SIZE
                        Batch size.
  --num_latent NUM_LATENT
                        Number of latent features.
  --l2_regularization L2_REGULARIZATION
                        L2 regularization coefficient.
  --weight_decay WEIGHT_DECAY
                        Weight decay coefficient.
  --neighborhood NEIGHBORHOOD
                        Neighborhood size for explainability.
  --top_k TOP_K         Cutoff k in MAP@k, HR@k and NDCG@k, etc.
  --lr LR               Learning rate.
  --optimizer OPTIMIZER
                        Optimizer: 'adam', 'sgd', 'rmsprop'.
  --sgd_momentum SGD_MOMENTUM
                        Momentum for SGD optimizer.
  --rmsprop_alpha RMSPROP_ALPHA
                        alpha hyperparameter for RMSProp optimizer.
  --rmsprop_momentum RMSPROP_MOMENTUM
                        Momentum for RMSProp optimizer.
  --loo_eval LOO_EVAL   True: LOO evaluation. False: Random train/test split
  --test_rate TEST_RATE
                        Test rate for random train/val/test split. test_rate
                        is the rate of test + validation. Used when 'loo_eval'
                        is set to False.
  --use_cuda USE_CUDA   True is you want to use a CUDA device.
  --device_id DEVICE_ID
                        ID of CUDA device if 'use_cuda' is True.
  --save_models SAVE_MODELS
                        True if you want to save the best model(s).
  --int_per_item INT_PER_ITEM
                        Minimum number of interactions per item for studying
                        effect sparsity on the lastfm-2k dataset.
```


* <b>Tune the hyperparameters of the models:</b>

```
python -m Code.hyperparameter_tuning [-h] [--model MODEL] [--dataset DATASET]
                                     [--num_configurations NUM_CONFIGURATIONS]
                                     [--num_reps NUM_REPS] [--num_epoch NUM_EPOCH]
                                     [--weight_decay WEIGHT_DECAY]
                                     [--neighborhood NEIGHBORHOOD] [--top_k TOP_K]
                                     [--lr LR] [--optimizer OPTIMIZER]
                                     [--sgd_momentum SGD_MOMENTUM]
                                     [--rmsprop_alpha RMSPROP_ALPHA]
                                     [--rmsprop_momentum RMSPROP_MOMENTUM]
                                     [--loo_eval LOO_EVAL] [--test_rate TEST_RATE]
                                     [--use_cuda USE_CUDA] [--device_id DEVICE_ID]
                                     [--save_models SAVE_MODELS]
                                     [--save_results SAVE_RESULTS]
                                     [--int_per_item INT_PER_ITEM]
```

Similarly, you can choose the dataset and the model. The code is set to perform a random hyperparameter tuning as presented in the paper. You can choose the number of experiments and replicates of each experiment. The hyperparameters tuned are the number of latent features, batch size and l2 regularization. The list of arguments is presented below:

```
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model to train: 'BPR', 'UBPR', 'EBPR', 'pUEBPR',
                        'UEBPR'.
  --dataset DATASET     'ml-100k' for Movielens 100K. 'ml-1m' for the
                        Movielens 1M dataset. 'lastfm-2k' for the Last.FM 2K
                        dataset. 'yahoo-r3' for the Yahoo! R3 dataset.
  --num_configurations NUM_CONFIGURATIONS
                        Number of random hyperparameter configurations.
  --num_reps NUM_REPS   Number of replicates per hyperparameter configuration.
  --num_epoch NUM_EPOCH
                        Number of training epochs.
  --weight_decay WEIGHT_DECAY
                        Weight decay coefficient.
  --neighborhood NEIGHBORHOOD
                        Neighborhood size for explainability.
  --top_k TOP_K         Cutoff k in MAP@k, HR@k and NDCG@k, etc.
  --lr LR               Learning rate.
  --optimizer OPTIMIZER
                        Optimizer: 'adam', 'sgd', 'rmsprop'.
  --sgd_momentum SGD_MOMENTUM
                        Momentum for SGD optimizer.
  --rmsprop_alpha RMSPROP_ALPHA
                        alpha hyperparameter for RMSProp optimizer.
  --rmsprop_momentum RMSPROP_MOMENTUM
                        Momentum for RMSProp optimizer.
  --loo_eval LOO_EVAL   True: LOO evaluation. False: Random train/test split
  --test_rate TEST_RATE
                        Test rate for random train/val/test split. test_rate
                        is the rate of test and validation. Used when
                        'loo_eval' is set to False.
  --use_cuda USE_CUDA   True if you want to use a CUDA device.
  --device_id DEVICE_ID
                        ID of CUDA device if 'use_cuda' is True.
  --save_models SAVE_MODELS
                        True if you want to save the best model(s).
  --save_results SAVE_RESULTS
                        True if you want to save the results in a csv file.
  --int_per_item INT_PER_ITEM
                        Minimum number of interactions per item for studying
                        effect sparsity on the lastfm-2k dataset.
```

## Datasets
We provide code ready to run on the:
* Movielens 100K dataset.
* Movielens 1M dataset.
* Last.FM 2K dataset.
* Yahoo! R3 dataset.

Note that, due to a consent that prevents us from sharing the Yahoo! R3 dataset, you need to download and add the dataset in a folder "Data/yahoo-r3" to be able to use it.

## Project structure
```
.
├── [ 67K]  Code
│   ├── [ 22K]  data.py
│   ├── [1.5K]  EBPR_model.py
│   ├── [ 12K]  engine_EBPR.py
│   ├── [ 11K]  hyperparameter_tuning.py
│   ├── [7.6K]  metrics.py
│   ├── [7.5K]  train_EBPR.py
│   └── [1.4K]  utils.py
├── [ 51M]  Data
│   ├── [ 12M]  lastfm-2k
│   │   ├── [1.8M]  artists.dat
│   │   ├── [4.4K]  readme.txt
│   │   ├── [217K]  tags.dat
│   │   ├── [1.1M]  user_artists.dat
│   │   ├── [221K]  user_friends.dat
│   │   ├── [4.0M]  user_taggedartists.dat
│   │   └── [4.8M]  user_taggedartists-timestamps.dat
│   ├── [ 15M]  ml-100k
│   │   ├── [ 716]  allbut.pl
│   │   ├── [ 643]  mku.sh
│   │   ├── [6.6K]  README
│   │   ├── [1.5M]  u1.base
│   │   ├── [383K]  u1.test
│   │   ├── [1.5M]  u2.base
│   │   ├── [386K]  u2.test
│   │   ├── [1.5M]  u3.base
│   │   ├── [387K]  u3.test
│   │   ├── [1.5M]  u4.base
│   │   ├── [388K]  u4.test
│   │   ├── [1.5M]  u5.base
│   │   ├── [388K]  u5.test
│   │   ├── [1.7M]  ua.base
│   │   ├── [182K]  ua.test
│   │   ├── [1.7M]  ub.base
│   │   ├── [182K]  ub.test
│   │   ├── [1.9M]  u.data
│   │   ├── [ 202]  u.genre
│   │   ├── [  36]  u.info
│   │   ├── [231K]  u.item
│   │   ├── [ 193]  u.occupation
│   │   └── [ 22K]  u.user
│   └── [ 24M]  ml-1m
│       ├── [167K]  movies.dat
│       ├── [ 23M]  ratings.dat
│       ├── [5.4K]  README
│       └── [131K]  users.dat
├── [ 30K]  images
│   └── [ 26K]  process_flow.svg
├── [ 34K]  LICENSE
├── [133K]  nbs
│   └── [129K]  P394476_Training_Explainable_BPR_model_on_LastFM_dataset.ipynb
├── [8.0K]  Output
│   └── [4.0K]  checkpoints
├── [9.4K]  README.md
└── [ 11K]  reports
    └── [7.0K]  S241566_report.ipynb

  52M used in 10 directories, 46 files
```

## References
[1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).<br>
[2] Saito, Yuta. "Unbiased Pairwise Learning from Implicit Feedback." NeurIPS 2019 Workshop on Causal Machine Learning. 2019.
