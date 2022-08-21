# Run OTM

OTM has two major steps:

+ Train deep model
+ Tree construction

The various generated files during these steps will be located in the paths specified in [`otm.conf`](https://github.com/massquantity/dismember/blob/main/configs/otm.conf) files.

The first step is training the deep model, and this step also includes the tree initialization:

```shell
$ ./tasks/bin/otm-train-deep-model --otmConfFile configs/otm.conf
```

The second step is constructing the tree: 

```shell
$ ./tasks/bin/otm-construct-tree --otmConfFile configs/otm.conf
```

The rest part is re-training the deep model and then re-constructing the tree. **But before that, modify the `initialize_mapping` parameter in `otm.conf` to `false`, so that the model will load the learned tree mapping in the second step**.