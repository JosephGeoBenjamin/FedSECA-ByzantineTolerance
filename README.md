# FedRISE
Offical Implementation of Paper "**FedSECA: Sign Election and Coordinate-wise Aggregation of Gradients
for Byzantine Tolerant Federated Learning**"

**Authors:** Joseph Geo Benjamin, Mothilal Asokan, Mohammad Yaqub, Karthik Nandakumar.

## Abstract

> One of the most common defense strategies against Byzantine clients in federated learning (FL) is to employ a robust aggregator mechanism that makes the training more resilient. While many existing Byzantine robust aggregators provide theoretical convergence guarantees and are empirically effective against certain categories of attacks, we observe that certain high-strength attacks can subvert the robust aggregator and collapse the training. To overcome this limitation, we propose a method called FedSECA for robust `S`ign `E`lection and `C`oordinate-wise `A`ggregation of gradients in FL that is less susceptible to malicious updates by an omniscient attacker. The proposed method has two main components. The **Concordance Ratio Induced Sign Election** (CRISE) module determines the consensus direction (elected sign) for each individual parameter gradient through a weighted voting strategy. The client weights are assigned based on a novel metric called concordance ratio, which quantifies the degree of sign agreement between the client gradient updates. Based on the elected sign, a **Robust Coordinate-wise Aggregation** (RoCA) strategy is employed, where variance-reduced sparse gradients are aggregated only if they are in alignment with the corresponding elected sign. We compare our proposed FedSECA method against 8 robust aggregators under 6 Byzantine attacks on 3 datasets and architectures. The results show that existing robust aggregators fail for at least some attacks, while FedSECA exhibits better robustness.

![FedSECA-Method](https://github.com/JosephGeoBenjamin/FedSECA-ByzantineTolerance/releases/download/cvpr25-v1/xFedSECA_method.png)

## Code base

With all possible abuse of SW-dev practices, the code is intentionally kept simple to ensure easier understanding, maintainability, and reduce the likelihood unintended behavior.

Current code base is intended for evaluating Cross-Silo use case. Same Global model is broadcasted to all clients after agregation is used for evaluation, not intended for personalization/fairness approaches that sends different models for each clients.

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
Dataset splits used for training, NoteBooks for plotting and all config files are available in the Downloads of [CVPR2025 releases](https://github.com/JosephGeoBenjamin/FedRISE-ByzantineTolerance/releases/tag/cvpr25-v1)

#### Possible Code Improvements:
(for future self or others)
1. Add multi-threading support to training multiple models in parallel, currently traiing happens sequentially
2. Add support for loading and unloading models to disk for each round, to support experiments in cross-device setting with thousands of models without needing to fit all in GPU(s) similtaneously.


## Cite Us

If you find this codebase useful for Model Poisoning or General FL experimentations, please consider giving a ⭐ for this Repo.

AND/OR

If you find our work on Byzantine Tolerance insightful for your research, consider citing us:

```

```


This work builds upon and enhances the FedRISE method developed during my Master’s thesis.

```
@article{benjamin2024byzantine,
  title={Byzantine Tolerant Gradient Aggregation for Cross-Silo Federated Learning},
  author={Benjamin, Joseph},
  year={2024}
}
```

More descriptive methods representation

![FedRISE_V2-method](https://github.com/JosephGeoBenjamin/FedSECA-ByzantineTolerance/releases/download/cvpr25-v1/xFedRISEV2-methods.png)