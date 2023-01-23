# Dismember<sup><a href="#fn1" id="ref1">1</a></sup>: Retrieval Algorithms for Decomposing Large-Scale Item Set

This repository implements a series of algorithms that aim to tackle the large-scale retrieval problem.

Due to the common situation that modern industrial applications such as online advertising, searching and recommendation usually contain a large number of items, complex deep learning models are often impractical to use directly. Since it requires traversing all the items to compute scores respectively, which leads to linear complexity w.r.t. the whole item set. To alleviate this issue, nowadays most real-world systems tend to convert it into an approximate similarity-searching(computing) problem, but the downside is the model expressiveness is limited.

Thus, the most intriguing feature of these implemented algorithms is providing logarithmic or sub-linear complexity w.r.t. the whole item set for model serving, where arbitrary deep models can be used. By introducing some kind of index structures, large-scale item set is decomposed into pieces, and relevant items can be retrieved through beam search.

Currently, the repository contains the following modules:

+ **scalann** is responsible for general deep learning model definition.
+ **tdm** implements the algorithm in [[Learning Tree-based Deep Model for Recommender Systems](https://arxiv.org/pdf/1801.02294.pdf)].
+ **jtm** implements the algorithm in [[Joint Optimization of Tree-based Index and Deep Model for Recommender Systems](https://arxiv.org/pdf/1902.07565.pdf)].
+ **otm** implements the algorithm in [[Learning Optimal Tree Models under Beam Search](https://arxiv.org/pdf/2006.15408.pdf)].
+ **deep-retrieval** implements the algorithm in [[Deep Retrieval: Learning A Retrievable Structure for Large-Scale Recommendations](https://arxiv.org/abs/2007.07203)].
+ **examples** contains some runnable tasks for these algorithms.

> <sup id="fn1">[1] Name comes from Swedish death metal band [Dismember](https://en.wikipedia.org/wiki/Dismember_(band)) <a href="#ref1" title="Jump back to footnote 1 in the text.">↩</a></sup>

## Requirements

+ Linux platform
+ Scala 2.13
+ [sbt](https://www.scala-sbt.org/) ~= 1.8.2

## Data

An example data is included in `data` folder, which comes from `movielens-1m` dataset.

## Build & Usage

```shell
$ git clone https://github.com/massquantity/dismember.git
```

For quick running, one can directly import it into an IDE, i.e. `IntelliJ IDEA` as described in [JetBrains doc](https://www.jetbrains.com/help/idea/sbt-support.html#import_sbt). We can also use sbt and [sbt-native-packager](https://sbt-native-packager.readthedocs.io/en/latest/index.html) to build packages and generate runnable scripts:

```shell
$ cd dismember
$ sbt Universal/packageZipTarball
```

The command will create a package called `examples-x.x.x.tgz` in `dismember/examples/target/universal/` folder. Now let's extract it into another folder, e.g. `tasks/`:

```shell
$ mkdir tasks
$ tar -zxf examples/target/universal/examples-0.1.0.tgz -C tasks --strip-components 1
```

At this point the `dismember/tasks/` folder should look like this:

```
bin/
  <bash script>
lib/
  <jar files>
```

Before running those tasks, one should set up the parameters in `xxx.conf` files first, which are located in `dismember/configs` folder. The descriptions of parameters are listed in [Configuration doc](https://github.com/massquantity/dismember/blob/main/doc/configuration.md).

Then head to the corresponding docs to see how to run these scripts:

+ [TDM](https://github.com/massquantity/dismember/blob/main/doc/TDM.md)
+ [JTM](https://github.com/massquantity/dismember/blob/main/doc/JTM.md)
+ [OTM](https://github.com/massquantity/dismember/blob/main/doc/OTM.md)
+ [Deep-Retrieval](https://github.com/massquantity/dismember/blob/main/doc/Deep-Retrieval.md)



## License

#### BSD-3-Clause
