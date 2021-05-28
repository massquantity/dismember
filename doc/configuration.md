# Configuration

For spark related configurations, see [Spark Configuration](https://spark.apache.org/docs/2.4.5/configuration.html).



## Initialize tree configs

| Entry                   | Example                              | Description                                                  |
| :---------------------- | :----------------------------------- | :----------------------------------------------------------- |
| init.seq_len            | 20                                   | Length of item sequence previously consumed by a user.       |
| init.min_seq_len        | 2                                    | Minimum length of item sequence consumed by a user.          |
| init.split_for_eval     | true                                 | Whether to split the whole data into train and eval data.    |
| init.split_ratio        | 0.8                                  | Set the ratio of data to split for eval data.                |
| init.data_path          | hdfs://user/hadoop/data.csv          | Original data path, either at local or HDFS.                 |
| init.train_path         | hdfs://user/hadoop/train.csv         | The generated train data path.                               |
| init.eval_path          | hdfs://user/hadoop/eval.csv          | The generated eval data path.                                |
| init.stat_path          | hdfs://user/hadoop/stat.txt          | The generated statistics data path, which records item frequency in data. |
| init.leaf_id_path       | hdfs://user/hadoop/leaf_id_data.txt  | The generated leaf id path, which records item ids.          |
| init.tree_protobuf_path | hdfs://user/hadoop/tree_pb_data.txt  | The generated tree file path, which saves the tree meta information in protobuf format. |
| init.user_consumed_path | hdfs://user/hadoop/user_consumed.txt | The generated user consumed path, which saves items consumed by a user in train data. |

## Train model configs

| Entry                          | Example                               | Description                                                  |
| ------------------------------ | ------------------------------------- | ------------------------------------------------------------ |
| model.deep_model               | DIN                                   | Deep model type in TDM, currently either be "DIN" or "DeepFM". |
| model.padding_index            | -1                                    | The specified index will be padded zero in Embedding_lookup and masked in Attention module. |
| model.train_path               | hdfs://user/hadoop/train_data.csv     | The train data path.                                         |
| model.eval_path                | hdfs://user/hadoop/eval_data.csv      | The eval data path.                                          |
| model.tree_protobuf_path       | hdfs://user/hadoop/tree_pb_data.txt   | Tree protobuf file path.                                     |
| model.evaluate_during_training | true                                  | Whether to evaluate during training.                         |
| model.thread_number            | 1                                     | Total CPU cores to use. 0 means using all available cores. This parameter is used in local mode only. |
| model.total_batch_size         | 2048                                  | Total train batch size summed up in all nodes and all cores. Note that TDM will sample negative items in all tree levels, which typically results in 50-100 negative items for one positive item. So this number means batch size after sampling. |
| model.total_eval_batch_size    | 2048                                  | Total eval batch size summed up in all nodes and all cores.  |
| model.seq_len                  | 20                                    | Length of item sequence consumed by a user.                  |
| model.layer_negative_counts    | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 | Number of negative items to sample per tree level. Since the tree root only has one node, the first number is always 0. |
| model.sample_with_probability  | true                                  | Whether to consider item frequency when sampling negative items. |
| model.start_sample_layer       | -1                                    | Set which level to start negative sampling.                  |
| model.sample_tolerance         | 20                                    | The model will at first try to sample different items for each level. But after exceeding the sample_tolerance, it may sample duplicate items. |
| model.parallel_sample          | true                                  | Whether to sample negative items for each batch in parallel. |
| model.embed_size               | 100                                   | Item embedding layer size.                                   |
| model.learning_rate            | 3e-4                                  | Training learning rate.                                      |
| model.iteration_number         | 100                                   | Total number of iterations for model training.               |
| model.show_progress_interval   | 10                                    | Interval of displaying evaluation result. For example, 10 means every 10 iteration, and value that <= 0 means every epoch. |
| model.topk_number              | 7                                     | Number of recommending items for a user. This parameter is used for evaluating. |
| model.candidate_num_per_layer  | 20                                    | Number of candidate items in each tree level. This parameter is used for evaluating. |
| model.model_path               | hdfs://user/hadoop/model.bin          | Path to save the model.                                      |
| model.embed_path               | hdfs://user/hadoop/embed.csv          | Path to save the embeddings. The embeddings will be  used for clustering. |

## Cluster tree configs

| Entry                      | Example                             | Description                                                  |
| -------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| cluster.embed_path         | hdfs://user/hadoop/embed.csv        | The embedding file path.                                     |
| cluster.tree_protobuf_path | hdfs://user/hadoop/tree_pb_data.txt | The generated tree file path, which saves the tree meta information in protobuf format. |
| cluster.parallel           | true                                | Whether to cluster the embeddings in parallel.               |
| cluster.thread_number      | 1                                   | Total CPU cores to use. 0 means using all available cores.   |