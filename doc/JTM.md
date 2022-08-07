# Run JTM

JTM has three major steps:

+ Initialize tree
+ Train deep model
+ Tree learning

The first two steps are the same as in TDM, since JTM only replaces the cluster tree part in TDM with the tree learning algorithm.

The various generated files during these steps will be located in the paths specified in `jtm.conf` files.

The first step is Initializing tree:

```shell
$ ./tasks/bin/jtm-initialize-tree --jtmConfFile configs/jtm.conf
```

The second step is training the deep model:

```shell
$ ./tasks/bin/jtm-train-deep-model --jtmConfFile configs/jtm.conf
```

The third step is learning the tree:

```shell
$ ./tasks/bin/jtm-tree-learning --jtmConfFile configs/jtm.conf
```

The rest part is re-training the deep model and then re-learning the tree, which are the same as the second and third step.