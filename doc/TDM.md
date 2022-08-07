# Run TDM

TDM has three major steps:

+ Initialize tree
+ Train deep model
+ Cluster tree

The various generated files during these steps will be located in the paths specified in `tdm.conf` files.

The first step is Initializing tree:

```shell
$ ./tasks/bin/tdm-initialize-tree --tdmConfFile configs/tdm.conf
```

The second step is training the deep model:

```shell
$ ./tasks/bin/tdm-train-deep-model --tdmConfFile configs/tdm.conf
```

The third step is clustering tree:

```shell
$ ./tasks/bin/tdm-cluster-tree --tdmConfFile configs/tdm.conf
```

The rest part is re-training the deep model and then re-clustering the tree, which are the same as the second and third step.
