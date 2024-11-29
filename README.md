# FedRISE
Offical Implementation of Paper "**FedRISE: Rating Induced Sign Election of Gradients for Byzantine Tolerant Federated Aggregation**"

**Authors:** Joseph Geo Benjamin, Mothilal Asokan, Mohammad Yaqub, Karthik Nandakumar.

## Abstract

> One of the most common defense strategies against model poisoning in federated learning is to employ a robust aggregator mechanism that makes the training more resilient.
Many of the existing Byzantine robust aggregators provide theoretical guarantees and are empirically effective against certain categories of attacks.
However, we observe that certain high-strength attacks can subvert the aggregator and collapse the training.
In addition, most aggregators require identifying tolerant settings to converge with considerable data heterogeneity, making aggregation extremely vulnerable.
Impact of attacks becomes more pronounced when the number of Byzantines is **near-majority**, and becomes harder to evade if the attacker is **omniscient** with access to data, honest updates and aggregation methods.
Motivated by these observations, we develop a robust aggregator called FedRISE for cross-silo FL that is consistent and less susceptible to poisoning updates by an omniscient attacker. The proposed method explicitly determines the optimal direction of each gradient through a sign-voting strategy that uses variance-reduced sparse gradients.
We argue that vote weighting based on the cosine similarity of raw gradients is misleading, and we introduce a sign-based gradient valuation function that ignores the gradient magnitude.
We compare our method against 8 robust aggregators under 6 poisoning attacks on 3 datasets and architectures. Our results show that existing robust aggregators collapse for at least some attacks under severe settings, while FedRISE demonstrates better robustness because of a stringent gradient inclusion formulation.


## Code base

With all possible abuse of SW-dev practices, the code is intentionally kept simple to ensure easier understanding, maintainability, and reduce the likelihood unintended behavior.

Current code base is intended for evaluating Cross-Silo use case. Same Global model is broadcasted to all clients after agregation is used for evaluation, not intended fairness based differnt models for each clients.

1. Follow installation steps in `setup.bash` for dependency.

2. Edit the config JSON in `configs` folder as need for specific for dataset, attack and defense settings

3. Run the following from base folder
    ```
    CUDA_VISIBLE_DEVICES=0 python tasks/cls-fedbase-train.py --load-json configs/cifar10-cls-fedByz.json
    ```
    here, setting `CUDA_VISIBLE_DEVICES=0,1,2,3` , will fill the FL models evenly in multiple GPUs memory, but training will only happen sequentially i.e train one model after other. No multithreading implemented, inorder to keep training codes beginer friendly.

4. Please read through train file `cls-fedbase-train.py` before running to understand the training setup. Most part are self-explanatory with additional comments as necessary.

5. The `algorithms` folder has all necessary buildingn blocks. The attacks are in `byznatine_attacks.py` and defense are in `federation_byz.py`. They are modular enough to include any new methods by creatin new class and including the class name in config file.

6. The `datacode` folder has all dataset class implementations. Please download and process the datasets from appropriate sources.

#### Downloads:
Dataset splits used for training, NoteBooks for plotting and all config files are available in the Downloads of [WACV25 releases](https://github.com/JosephGeoBenjamin/FedRISE-ByzantineTolerance/releases/tag/wacv25-v1)

#### Possible Code Improvements:
(for future self or others)
1. Add multi-threading support to training multiple models in parallel
2. Add support for loading and unloading models to disk for each round, to support experiments in cross-device setting with thousands of models without needing to fit all in GPU(s) similtaneously.


## Cite Us

If you find this codebase useful for Model Poisoning or General FL experimentations, please consider giving a ⭐ for this Repo.

AND/OR

If you find our work on Byzantine Tolerance insightful for your research, consider citing us:

```

```
