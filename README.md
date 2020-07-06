# Cats or CAT Scans 



# Background

Networks for medical image classification can be pretrained on a variety of datasets, from natural images such as cats, to other medical image datasets such as CT scans. There is no consensus on what strategy is best, an overview of several papers which investigate this is here: https://arxiv.org/abs/1810.05444 . 
Two factors seem to be important:

1) Source data needs to be "large enough" 
2) Source data needs to be "similar" to target data

The exact definitions of these factors are not given, nor is their relative importance. This is what we want to investigate in this project. 

# Experiments

## Dataset size 
* Select data sources S1-S5 (for example)
* Train networks on different subset sizes of S1-S5
* Test on target data T1-T5

## Dataset similarity
* Define multiple ways to measure similarity
* Compare similarity embedding to "ground truth" from previous experiment 

# Related repositories

https://github.com/tueimage/transfer-medical-bsc-2018
https://github.com/tueimage/transfer-medical-msc-2019





