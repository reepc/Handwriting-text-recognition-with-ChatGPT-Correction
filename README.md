# Introduction

This repo use [TrOCR model](https://github.com/microsoft/unilm/tree/master/trocr#trocr), which is from meta.
You can see more details there.

The online version will be upload to [NTUST NLPLab's website](https://nlp.csie.ntust.edu.tw/), which a laboratory in Taiwan.

The version here is which you can run the model locally (in your PC).
If you want to run in your PC, you need a GPU which CUDA supports (Nvidia's GPU).
Or you can run in kaggle or colab, which are two cloud platforms.

## Prerequisites

### Anaconda (Not necessary)

You can find the installing steps [here](https://docs.anaconda.com/free/anaconda/install/#)

### Install python (without anaconda)

If you have python version below, you can skip this step.

- 3.8
- 3.9
- 3.10
- 3.11

These python versions can install pytorch.
If you still have errors because of python version, please install python 3.9.13, which is the author's python version when doing this repo.

#### Windows

You can find python version [here](https://www.python.org/downloads/)
Please click "Add python.exe to PATH" when installing.

#### Linux

```
$sudo apt-get intall python3.9.13
```
