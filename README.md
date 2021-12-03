# CAPGNN
Source code and dataset of the paper ["Contrastive Adaptive Propagation Graph Neural Networks forEfficient Graph Learning"](https://arxiv.org/abs/2112.01110)


Paper URL: [https://arxiv.org/abs/2112.01110](https://arxiv.org/abs/2112.01110)


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


## Cite

```html
@misc{hu2021contrastive,
      title={Contrastive Adaptive Propagation Graph Neural Networks for Efficient Graph Learning}, 
      author={Jun Hu and Shengsheng Qian and Quan Fang and Changsheng Xu},
      year={2021},
      eprint={2112.01110},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
