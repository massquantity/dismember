# Configuration

**Note that in `xxx.conf` files, same parameters should have consistent values in different steps. For example in `tdm.conf`, value of `init.seq_len` should be equal to `model.seq_len`.**

Also, it is recommended to use absolute path in `xxx_path` parameters.

## TDM - Initialize tree configs

| Entry                   | Example                          | Description                                                  |
| :---------------------- | :------------------------------- | :----------------------------------------------------------- |
| init.seq_len            | 10                               | Length of item sequence previously consumed by a user.       |
| init.min_seq_len        | 2                                | Minimum length of item sequence consumed by a user.          |
| init.split_for_eval     | true                             | Whether to split the whole data into train and eval data.    |
| init.split_ratio        | 0.8                              | Set the ratio of data to split for eval data.                |
| init.data_path          | dismember/data/example_data.csv  | Original data path.                                          |
| init.train_path         | dismember/data/train_data.csv    | The generated train data path.                               |
| init.eval_path          | dismember/data/eval_data.csv     | The generated evaluate data path.                            |
| init.stat_path          | dismember/data/stat_data.txt     | The generated statistics data path, which records item frequency in data. |
| init.leaf_id_path       | dismember/data/leaf_id_data.txt  | The generated leaf id path, which records item ids.          |
| init.tree_protobuf_path | dismember/data/tdm_tree.bin      | The generated tree file path, which saves the tree meta information in protobuf format. |
| init.user_consumed_path | dismember/data/user_consumed.txt | The generated user consumed path, which saves items consumed by a user in train data. |

## TDM - Train model configs

| Entry                          | Example                              | Description                                                                                                                                                                                                                                       |
|--------------------------------| ------------------------------------ |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model.deep_model               | DIN                                  | Deep model type in TDM, currently either be "DIN" or "DeepFM".                                                                                                                                                                                    |
| model.train_path               | dismember/data/train_data.csv | The train data path.                                                                                                                                                                                                                              |
| model.eval_path                | dismember/data/eval_data.csv | The evaluate data path.                                                                                                                                                                                                                           |
| model.tree_protobuf_path       | dismember/data/tdm_tree.bin | Tree protobuf file path.                                                                                                                                                                                                                          |
| model.evaluate_during_training | true                                 | Whether to evaluate during training.                                                                                                                                                                                                              |
| model.thread_number            | 1                                    | Total CPU cores to use. 0 means using all available cores.                                                                                                                                             |
| model.total_batch_size         | 2048                                 | Total train batch size summed up in all cores. Note that TDM will sample negative items in all tree levels, which typically results in 50-100 negative items for one positive item. So this number means batch size after sampling. |
| model.total_eval_batch_size    | 2048                                 | Total evaluate batch size summed up in all cores.                                                                                                                                                                                   |
| model.seq_len                  | 10                                  | Length of item sequence consumed by a user.                                                                                                                                                                                                       |
| model.layer_negative_counts    | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 | Number of negative items to sample per tree level. Since the tree root only has one node, the first number is always 0.                                                                                                                           |
| model.sample_with_probability  | false                             | Whether to consider item frequency when sampling negative items.                                                                                                                                                                                  |
| model.start_sample_level       | 1                                    | Set which level to start negative sampling, which should be larger than 0 since the only root node should be excluded.                                                                                                                            |
| model.sample_tolerance         | 20                                   | The model will at first try to sample different items for each level. But after exceeding the `sample_tolerance`, it may sample duplicate items.                                                                                                  |
| model.parallel_sample          | true                                 | Whether to sample negative items for each batch in parallel.                                                                                                                                                                                      |
| model.embed_size               | 16                                 | Item embedding layer size.                                                                                                                                                                                                                        |
| model.learning_rate            | 3e-4                                 | Training learning rate.                                                                                                                                                                                                                           |
| model.iteration_number         | 100                                  | Total number of iterations for model training.                                                                                                                                                                                                    |
| model.show_progress_interval   | 10                                   | Interval of displaying evaluation result. For example, 10 means every 10 iteration, and value that <= 0 means every epoch.                                                                                                                        |
| model.topk_number              | 7                                    | Number of recommending items for a user. This parameter is used for evaluating.                                                                                                                                                                   |
| model.beam_size                | 20                                   | Number of beam search candidate nodes in each tree level. This parameter is used for evaluating.                                                                                                                                             |
| model.model_path               | dismember/data/tdm_model.bin | Path to save the model.                                                                                                                                                                                                                           |
| model.embed_path               | dismember/data/embed.csv | Path to save the embeddings. The embeddings will be  used for clustering tree.                                                                                                                                                                    |

## TDM - Cluster tree configs

| Entry                      | Example                     | Description                                                  |
| -------------------------- | --------------------------- | ------------------------------------------------------------ |
| cluster.embed_path         | dismember/data/embed.csv    | The embedding file path.                                     |
| cluster.tree_protobuf_path | dismember/data/tdm_tree.bin | The generated tree file path, which saves the tree meta information in protobuf format. |
| cluster.cluster_type       | kmeans                      | clustering algorithm, either be "kmeans" or "spectral".      |
| cluster.cluster_iter       | 10                          | clustering number of iterations.                             |
| cluster.parallel           | true                        | Whether to cluster the embeddings in parallel, note that spectral clustering doesn't support parallel mode. |
| cluster.thread_number      | 1                           | Total CPU cores to use. 0 means using all available cores.   |

<br>

## JTM - Initialize tree configs

Same as `TDM - Initialize tree configs`.

## JTM - Train model configs

Same as `TDM - Train model configs.`

## JTM - Tree learning configs

| Entry                        | Example                         | Description                                                  |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------ |
| tree.data_path               | dismember/data/example_data.csv | Original data path.                                          |
| tree.model_path              | dismember/data/jtm_model.bin    | Path to load the model.                                      |
| tree.tree_protobuf_path      | dismember/data/jtm_tree.bin     | Tree protobuf file path.                                     |
| tree.deep_model              | DIN                             | Deep model type in JTM, currently either be "DIN" or "DeepFM". |
| tree.gap                     | 2                               | Tree level gap in item assignment task. It's the hyper-parameter "d" in the paper. |
| tree.seq_len                 | 10                              | Length of item sequence previously consumed by a user.       |
| tree.hierarchical_preference | false                           | Whether to use "hierarchical user preference representation" described in section 3.3 of the paper. |
| tree.min_level               | 0                               | Minimal tree level to use hierarchical preference. If target level < min_level, the original item node instead of its ancestor will be used in item sequence. Since low level ancestors such as root node may not be a good representation for all items. |
| tree.thread_number           | 1                               | Total CPU cores to use. 0 means using all available cores.   |

<br>

## OTM - Train model configs

| Entry                        | Example                         | Description                                                  |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------ |
| model.data_path              | dismember/data/example_data.csv | Original data path.                                          |
| model.model_path             | dismember/data/otm_model.bin    | Path to save the model.                                      |
| model.deep_model             | DIN                             | Deep model type in OTM, currently either be "DIN" or "DeepFM". |
| model.thread_number          | 1                               | Total CPU cores to use. 0 means using all available cores.   |
| model.train_batch_size       | 2048                            | Total train batch size summed up in all cores.               |
| model.eval_batch_size        | 2048                            | Total evaluate batch size summed up in all cores.            |
| model.embed_size             | 16                              | Item embedding layer size.                                   |
| model.learning_rate          | 3e-3                            | Training learning rate.                                      |
| model.epoch_num              | 1                               | Number of epoch for model training.                          |
| model.topk_number            | 7                               | Number of recommending items for a user. This parameter is used for evaluating. |
| model.beam_size              | 20                              | Number of beam search candidate nodes in each tree level. This parameter is used for evaluating. |
| model.show_progress_interval | 200                             | Interval of displaying evaluation result. For example, 10 means every 10 iteration, and value that <= 0 means every epoch. |
| model.seq_len                | 10                              | Length of item sequence previously consumed by a user.       |
| model.min_seq_len            | 2                               | Minimum length of item sequence consumed by a user.          |
| model.split_ratio            | 0.8                             | Set the ratio of data to split for eval data.                |
| model.leaf_init_mode         | random                          | Tree initialization mode, either be "random" or "category". "category" means using item category information as described in TDM paper. |
| model.initialize_mapping     | true                            | If set to true, will initialize tree according to "model.leaf_init_mode". If set to false, will load tree from "model.mapping_path". |
| model.mapping_path           | dismember/data/otm_mapping.txt  | File path for Item to leaf node mapping. Since according to the paper, a tree can be represented as item-node mapping, so this parameter can be also be considered as the tree file path. |
| model.label_num              | 5                               | Number of labels in one sample, since OTM supports multiple labels. |
| model.target_mode            | pseudo                          | Target construction mode, either be "pseudo" or "normal". "pseudo" denotes optimal pseudo target in the paper, whereas "normal" denotes "OTM(-OptEst)" described in experiments of the paper. |
| model.seed                   | 42                              | Seed used in random tree initialize initialization.          |

## OTM - Construct tree configs

Note: OTM uses same mechanism for tree construction as in JTM, so parameters are similar.

| Entry              | Example                         | Description                                                  |
| ------------------ | ------------------------------- | ------------------------------------------------------------ |
| tree.data_path     | dismember/data/example_data.csv | Original data path.                                          |
| tree.model_path    | dismember/data/otm_model.bin    | Path to load the model.                                      |
| tree.mapping_path  | dismember/data/otm_mapping.txt  | Path to save and load tree and item-node mapping.            |
| tree.deep_model    | DIN                             | Deep model type in OTM, currently either be "DIN" or "DeepFM". |
| tree.gap           | 2                               | Tree level gap in item assignment task. It's the hyper-parameter "d" in JTM paper. |
| tree.label_num     | 5                               | Number of labels in one sample, since OTM supports multiple labels. |
| tree.seq_len       | 10                              | Length of item sequence previously consumed by a user.       |
| tree.min_seq_len   | 2                               | Minimum length of item sequence consumed by a user.          |
| tree.split_ratio   | 0.8                             | Set the ratio of data to split for eval data.                |
| tree.thread_number | 1                               | Total CPU cores to use. 0 means using all available cores.   |

<br>

## Deep Retrieval - Train model configs

| Entry                        | Example                         | Description                                                  |
| ---------------------------- | ------------------------------- | ------------------------------------------------------------ |
| model.data_path              | dismember/data/example_data.csv | Original data path.                                          |
| model.model_path             | dismember/data/dr_model.bin     | Path to save the model.                                      |
| model.mapping_path           | dismember/data/dr_mapping.txt   | Path to load item-path mapping.                              |
| model.thread_num             | 1                               | Total CPU cores to use. 0 means using all available cores.   |
| model.train_batch_size       | 2048                            | Total train batch size summed up in all cores.               |
| model.eval_batch_size        | 2048                            | Total evaluate batch size summed up in all cores.            |
| model.num_layer              | 3                               | Depth of model. It's the hyper-parameter "D" in the paper.   |
| model.num_node               | 100                             | Number of nodes in one layer. It's the hyper-parameter "K" in the paper. |
| model.num_path_per_item      | 2                               | Number of paths for one item. It's the hyper-parameter "J" in the paper. |
| model.embed_size             | 16                              | Item embedding layer size.                                   |
| model.learning_rate          | 3e-3                            | Training learning rate.                                      |
| model.epoch_num              | 1                               | Number of epoch for model training.                          |
| model.num_sampled            | 1                               | Number of negative items sampled for one positive item in sampled softmax. |
| model.topk_number            | 7                               | Number of recommending items for a user.                     |
| model.beam_size              | 20                              | Number of beam search candidate nodes in each layer.         |
| model.show_progress_interval | 100                             | Interval of displaying evaluation result. For example, 10 means every 10 iteration, and value that <= 0 means every epoch. |
| model.seq_len                | 10                              | Length of item sequence previously consumed by a user.       |
| model.min_seq_len            | 2                               | Minimum length of item sequence consumed by a user.          |
| model.split_ratio            | 0.8                             | Set the ratio of data to split for eval data.                |
| model.initialize_mapping     | true                            | If set to true, will initialize item-path mapping randomly. If set to false, will load mapping from "model.mapping_path". |

## Deep Retrieval - Coordinate Descent configs

| Entry                 | Example                         | Description                                                  |
| --------------------- | ------------------------------- | ------------------------------------------------------------ |
| cd.data_path          | dismember/data/example_data.csv | Original data path.                                          |
| cd.model_path         | dismember/data/dr_model.bin     | Path to save the model.                                      |
| cd.mapping_path       | dismember/data/dr_mapping.txt   | Path to load item-path mapping.                              |
| cd.thread_num         | 1                               | Total CPU cores to use. 0 means using all available cores.   |
| cd.train_batch_size   | 2048                            | Total train batch size summed up in all cores.               |
| cd.eval_batch_size    | 2048                            | Total evaluate batch size summed up in all cores.            |
| cd.num_layer          | 3                               | Depth of model. It's the hyper-parameter "D" in the paper.   |
| cd.num_node           | 100                             | Number of nodes in one layer. It's the hyper-parameter "K" in the paper. |
| cd.num_path_per_item  | 2                               | Number of paths for one item. It's the hyper-parameter "J" in the paper. |
| cd.seq_len            | 10                              | Length of item sequence previously consumed by a user.       |
| cd.min_seq_len        | 2                               | Minimum length of item sequence consumed by a user.          |
| cd.split_ratio        | 0.8                             | Set the ratio of data to split for eval data.                |
| cd.initialize_mapping | true                            | If set to true, will initialize item-path mapping randomly. If set to false, will load mapping from "cd.mapping_path". |
| cd.candidate_path_num | 20                              | Number of candidate paths computed before coordinate descent. It's the hyper-parameter "S" in the paper. |
| cd.iteration_num      | 3                               | Number of iterations in coordinate descent.                  |
| cd.decay_factor       | 0.999                           | Decay factor in streaming training. It's the hyper-parameter "η" in the paper. |
| cd.penalty_factor     | 3e-6                            | Penalty factor in coordinate descent. It's the hyper-parameter "α" in the paper. |
| cd.penalty_poly_order | 4                               | Degree of polynomials in penalty function.                   |
| cd.train_mode         | streaming                       | Method for computing s[v,c] in the paper, either be "batch" or "streaming". |

