# Run Deep Retrieval

Deep Retrieval has two major steps:

+ Train deep model
+ Update mapping by Coordinate Descent

The various generated files during these steps will be located in the paths specified in `deep-retrieval.conf` files.

The first step is training the deep model, and this step will also initialize the mapping:

```shell
$ ./tasks/bin/dr-train-deep-model --drConfFile configs/deep-retrieval.conf
```

The second step is coordinate descent: 

```shell
$ ./tasks/bin/dr-coordinate-descent --drConfFile configs/deep-retrieval.conf
```

The rest part is re-training the deep model and then re-updating the mapping. **But before that, modify the `initialize_mapping` parameter in `deep-retrieval.conf` to `false`, so that the model will load the learned path mapping in the second step**.