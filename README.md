# Tesi-2021-2022

## Installation
Use pip to install the required files:
```
pip install -r requirements.txt
pip install git+https://github.com/elastic/ember.git
pip install -e asm2vec-pytorch
```

## Get Started
There are different pipelines to classify malware based on which type of binary wanna classify:
- PE
- ELF

In the first case the pipeline should be builded like follows:
1. Create the dataset using the preprocessing module
3. Build a supervised learning algorithm using BODMAS pipeline
4. Classify 

If you are interested in classifying ELF binaries:
1. Use the asm2vec-pytorch module to build and train the NN
2. Extract the features with the script `extract_vectors.py`
3. Use the frequency_clusters module to classify the binaries
