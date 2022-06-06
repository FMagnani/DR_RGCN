# Drug Repurposing - Relational GCN

Relational Graph Convolutional Networks are a class of Graph Neural Networks suitable for working on heterogeneous (multiple node types) 
and/or multi-modal (multiple edge types) graphs. The reference paper is [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) and the reference implementation that I chose is [this DGL example](https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn).  
The benchmark dataset for this kind of architectures is the [FB15k-237](https://paperswithcode.com/dataset/fb15k). The main difference with respect to the typical Drug Repurposing knowledge 
graphs (as [DRKG](https://github.com/gnn4dr/DRKG)) is the size. With the code given here, you can experiment with various hyperparameter choices and any
dataset you like.  
  
## How to add a dataset
Input graphs are given in the _edge list_ format, i.e. it is needed:  
* Three files called `train.txt`, `test.txt`, `valid.txt` (that are tsv), which contain the triplets of the graph used for training, testing and validating respectively.  
* Two files called `entities.dict` and `relations.dict` (that are tsv), which contains simply the lists of the unique identifiers of the nodes and of the relations (all the edge types).  
  
An example of such files is given into the folder `DRKG`, with only 10 entries, just as a reference.  
All these files are into a folder, for example into `DRKG` or `FB15k-237`. The name of such folder is the argument "--dataset" that's given to the script 
`link.py`. The path to all these folders is the "raw_dir" variable that must be set in `link.py`. In practice, if we have two datasets into two folders 
`/home/user/projects/DRKG` and `/home/user/projects/FB15k-237`, "raw_dir" is "/home/user/projects" and the name of the dataset is given to `link.py` as an argument.  

## How to use the script
It's enough to  
1. Set the "raw_dir" variable in `link.py` (only once) and save the dataset into `raw_dir/name`
2. Run `python link.py --dataset name`

The other arguments of `link.py` are 
* gpu, that is 0 if there's only one gpu
* eval-protocol, that slightly changes the protocol for the metric computation
* edge-sampler, that can be "uniform" or "neighbor" (the latter gave less problems to me)
* test, which selects if performing or not the test on the test triplets. If the test is not done, you can still check the convergence through the training loss. 

## Hyperparameters
There are many hyperparameters that can be chosen.  
  
**R-GCN model**  
| Hyperparameter | Variable in the code | Meaning |  
|---|---|---|  
| Hidden Dimensions | _h_dim_ (in _LinkPredict_) | Dimensionality of the hidden space |  
| Type of regularization | _regularizer_ (in _LinkPredict_) | The R-GCN layer uses parameter regularization, that can either be: a basis decomposition (regularizer='basis') or a block diagonal decomposition ('bdd'). See [the paper](https://arxiv.org/abs/1703.06103) for details. |  
| Number of bases | _num_bases_ (in _LinkPredict_) | The number of the bases (if regularizer='basis') or the number or block diagonal matrices (if regularizer='bdd') used for regularizing the layers. |  
| Loss function | _predict_loss_ (in _LinkPredict.get_loss_) | A classification loss for link prediction. Among the most popular ones are the _cross entropy with logits_ and the _soft margin loss_. |  
  
**Training**
| Hyperparameter | Variable in the code | Meaning |  
|---|---|---|  
| Batch size | _sample_size_ (in _SubgraphIterator_) | Size of the batches of triplets used in training |  
| Number of epochs | _num_epochs_ (in _SubgraphIterator_) | The number of training epochs to perform. One epoch trains over a single batch of triplets. |  
| Learning Rate | _lr_ (in _optimizer_) | The learning rate of the Adam optimizer. It's the optimizer of the Graph Convolutional layers. In theory, the Graph Convolution is applied for embedding the entities, while there it is shallow embedding for the relations. But I have to understand yet where and how this is done in the code, precisely. |  

## Note on testing on large graphs
The training (in batches) is made on GPU, while the testing (not in batches) is made on the CPU.  
For large graphs (in practice the DRKG), I had memory problems (error for trying to allocate too much memory) with testing. I solved by not testing, and simply 
checking the convergence on the training set.  
In particular, the code that perform the testing on the test set of triplets is the if-block starting at line 114. Disabling that block, the training will still be made and 
the training loss will be computed. I suggest to perform the test with FB15k-237, but not with DRKG.  
The behaviour can be selected with the argument "--test" given to `link.py`.  
Also, see the next note.  

## Note on hardware compatiblity
The training part works on a wider range of systems, while the testing part of the code gave me an error:  
`dgl._ffi.base.DGLError: /opt/dgl/src/array/cpu/./spmm_blocking_libxsmm.h:267: Failed to generate libxsmm kernel for the SpMM operation!`  
which is solved working on a different hardware (precisely: tfm1 and tfm2 gave error, while the server of Pati did not).  
Yet, you can train the model without testing it. The convergence can be checked by the training loss.  
