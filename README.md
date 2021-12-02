# CAPGNN
Source code and dataset of the paper "Contrastive Adaptive Propagation Graph Neural Networks forEfficient Graph Learning"


## Requirements

+ Linux
+ Python 3.7
+ tensorflow == 2.4.1
+ tf_geometric == 0.0.65
+ tqdm=4.51.0


## Run CAPGNN

You can run CAPGNN (CAPGCN and CAPGAT) with the following command:
```shell
sh run_capgnn_${DATASET}.sh ${METHOD} ${GPUS}
```
where you should replace ${DATASET} with the dataset name, replace ${METHOD} with CAPGCN or CAPGAT, and replace ${GPUS} with the gpu ids.
The supported dataset names are listed as follows:
+ cora
+ citeseer
+ pubmed
+ amazon-computers
+ amazon-photo


For example, you can use the following command to run CAPGAT with GPU0 on the Cora dataset.
The command will run CAPGAT for multiple times and outputs the results in results.txt.

```shell
sh run_capgnn_cora.sh CAPGAT 0
```