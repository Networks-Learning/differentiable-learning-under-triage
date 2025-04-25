# Differentiable Learning Under Triage

This is a repository containing code and data for the paper:

> N. Okati, A. De, and M. Gomez-Rodriguez. _Differentiable Learning Under Triage_. Advances in Neural Information Processing Systems (NeurIPS), 2021.

The paper is published in NeurIPS'21 and can be found [here](https://proceedings.neurips.cc/paper/2021/file/4c4c937b67cc8d785cea1e42ccea185c-Paper.pdf).


## Pre-requisites

This code depends on the following packages:

 1. `jupyterlab`
 2. `Torch`
 3. `fasttext`
 4. `matplotlib`
 5. `numpy`
 


## Structure

 - `Hatespeech` contains the data and code for regenerating the experiments on [Hatespeech](https://github.com/t-davidson/hate-speech-and-offensive-language) dataset.
 - `Galaxy-zoo` contains the data and code for regenerating the experiments on [Galaxy-zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) dataset.
 - `Synthetic` contains the code for regenerating the experiments on Synthetic dataset.
 
Please refer to section 5 and 6 of [our paper](https://arxiv.org/pdf/2103.08902.pdf) for detailed discussion on experiments.


## Execution

The details for executing are mentioned inside each folder. Each folder contains a `train.ipynb` in which our algorithm and baselines are implemented and the results are illustrated.

## Citation
Please cite us if you use our work in your research:

```
@inproceedings{okati2021differentiable,
      title={Differentiable Learning Under Triage}, 
      author={Nastaran Okati and Abir De and Manuel Gomez-Rodriguez},
      booktitle={Advances in Neural Information Processing Systems},
      year={2021}
}
```

