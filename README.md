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
1. Use the asm2vec-pytorch module to build and train the NN following the well written [doc](https://github.com/CaptWake/Tesi-2021-2022/blob/main/asm2vec-pytorch/README.md)
2. Extract the features with the script `extract_vectors.py` 
```
$ python asm2vec-pytorch/scripts/extract_vectors.py -m mymodel.pt -o dataset.json 
```
3. You can use now the frequency_clusters module to classify the binaries specifying the configuration file
```
$ python unsupervised/frequency_clusters.py -cf example_setting.json
```
Where example_setting.json should be something like this:
```json
{
    "dataset": "/path/to/your/dataset",
    "output_path": "example/",
    "binary2class": "/path/to/binary2class file", // this file should be a json file containing association of the form 
                                                  // "filename" : "malwareclass"
    "model": {
        "name": "KMeans",
        "params": {
            "n_clusters": 7
        }
    }
}
```
> **Note**  
> If you want use the legacy mode you can omit the model specification in the configuration file 
## Extras
In case of supervised learning you could use the avclass package that helps you assign labels based on VT reports.  
A quick example helps illustrating the labeling process:  
With the help of the `vt_report_adapter.py` module you can generate a reports.json from all the vt reports inside the reports folder
```
$ python vt_report_adapter.py -i reports/
```
you can use this file as input for the `avclass2_labeler.py` module
```sh
$ python avclass/avclass2/avclass2_labeler.py -lb reports.json -p
```
the final output looks like this:
```
[-] 0 JSON readaca2d12934935b070df8f50e06a20539 75      CLASS:grayware|15,FILE:os:windows|13,CLASS:grayware:adware|11,FAM:adrotator|8
76c643bd32186c2c7cb1f52c38c07bb3        68      UNK:disabler|13,UNK:winreg|8,UNK:prova|4,FILE:os:windows|2
[-] 2 JSON read
[-] Samples: 2 NoScans: 0 NoTags: 0 GroundTruth: 0
```
