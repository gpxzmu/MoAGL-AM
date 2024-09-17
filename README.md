# MoAGL-SA: A Multi-Omics Adaptive Integration Method with Graph Learning and Self Attention for Cancer Subtype Classification

<sup>MoAGL-SA is an adaptive multi-omics integration method based on graph learning and self-attention. First, patient similarity graphs are generated from each omics dataset using graph learning. Next, three-layer graph convolutional networks are employed to extract omic-specific graph embeddings. Self-attention is then used to focus on the most relevant omics, adaptively assigning weights to different graph embeddings for multi-omics integration. Finally, cancer subtypes are classified using a softmax classifier.<sup>

## Requirements

+ python = 3.8.10
+ torch = 1.11.0+cu113
+ pandas = 1.5.2
+ numpy = 1.23.5
+ sklearn = 1.1.3

~~~python
# It is recommended to use the conda command to configure the environment:
conda env create -f environment.yml
~~~

MoAGL-SA is based on the Python program language.  

## Files

data: Input multi-omics data

*MoAGL-SA.py*: Examples of MoAGL-SA

*models.py*: MoAGL-SA key module

*train_test.py*: Training and testing functions

*main_biomarker.py*: Examples for identifying biomarkers

*feat_importance.py*: Feature importance functions

*utils.py*: Supporting functions    



