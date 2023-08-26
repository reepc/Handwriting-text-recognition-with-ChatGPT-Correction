# Introduction

This repo use [TrOCR model](https://github.com/microsoft/unilm/tree/master/trocr#trocr), which is from meta.
You can see more details there.

The online version will be upload to [NTUST NLPLab's website](https://nlp.csie.ntust.edu.tw/).

The version here is which you can run the model locally (in your PC).
If you want to run in your PC, you need a GPU which CUDA supports (Nvidia's GPU).
Or you can run in kaggle or colab, they both have some free compute resource.

You can train an adapter according to your font to increase model's accuracy.

## Prerequisites

### Anaconda (Not necessary)

You can find the installing steps [here](https://docs.anaconda.com/free/anaconda/install/#)

#### Create anaconda environment (Same as Install Python)

Open your anaconda prompt (Windows) or shell (Linux) and type:
```
$ conda create -n the_name_you_want python=3.9.13
```

### Install Python (without anaconda)

If you have python version below, you can skip this step.

- 3.8
- 3.9
- 3.10
- 3.11

These python versions can install pytorch.
If you still have errors because of python version, please install python 3.9.13, which is the author's python version when doing this repo.

#### Windows

You can find python versions [here](https://www.python.org/downloads/)

Please click "Add python.exe to PATH" when installing.

#### Linux

```
$ sudo apt-get update 
$ sudo apt-get upgrade
$ sudo apt-get intall python=3.9.13
```

#### MacOS

Also, find python versions [here](https://www.python.org/downloads/macos/)

## Installation

```
$ git clone https://github.com/reepc/Handwriting-text-recognition-with-ChatGPT-Correction.git
$ cd Handwriting-text-recognition-with-ChatGPT-Correction
$ pip install -r requirements.txt
```

## Adapter training and evalution

### Adapter training


If you have any problem, question or want to share your using experience, feel free to contact with guwanjun0530@outlook.com