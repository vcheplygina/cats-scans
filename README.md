# Cats or CAT Scans 


Networks for medical image classification can be pretrained on a variety of datasets, from natural images such as cats, to other medical image datasets such as CT scans. 
There is no consensus on what strategy is best, an overview of several papers which investigate this is here: https://arxiv.org/abs/1810.05444 . 
Two factors seem to be important:

1) Source data needs to be "large enough" 
2) Source data needs to be "similar" to target data

The exact definitions of these factors are not given, nor is their relative importance. 

Some preliminary results are available in the thesis of Floris Fok (code https://github.com/tueimage/transfer-medical-bsc-2018, report available as PDF):

* Source datasets used are ImageNet (1M images), CatDog (25K), Natural (8K), Chest (6K), DiabeticRetinopathy (35K) 
* Target datasets used are Chest (6K), ISIC 2017 (2K), BloodCell (12.5K)
* Averaged over different target datasets, ImageNet is the best source, and DiabeticRetinopathy is the second best 
* Transfer learning with fine-tuning outperforms transfer learning by feature extraction 


# Project

This project consists of two parts: 

* Establishing "transferability" of datasets by training on different source datasets, and testing on different targets.
* Defining ways to measure the similarity of datasets, such that the similarity reflects the transferability

## Part 1 - Transfer Learning 
The project includes nine different datasets created from six databases. These are the following:
1. [ImageNet](http://image-net.org)
2. [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
3. [ISIC2018 - Task 3 - the training set](https://challenge2018.isic-archive.com/task3/training/)
4. [Chest X-rays](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
5. [PatchCamelyon (PCam)](http://basveeling.nl/posts/pcam/)
6. [KimiaPath960](https://www.kaggle.com/ambarish/kimia-path-960)

Two ImageNet subsets, STL-10 and self-made subset STI-10, are used. To speed up training PCam is decreased in size to 
100.000 and 10.000 for subsets PCam-middle and PCam-small.

The nine different datasets are all used as source dataset, the medical datasets ISIC2018, Chest X-rays and PCam-middle 
are used as target dataset.


### Built With

* Keras
* Sacred
* Neptune 
* OSF

The experiments are logged used Sacred with a Neptune observer. The results, code version and model settings of all
experiments can be found on https://ui.neptune.ai/irmavdbrandt/cats-scans/experiments?viewId=standard-view. For access
please e-mail to irma.vdbrandt@gmail.com. 

The trained models and predictions are stored on Open Science Framework: 
https://osf.io/x2fpg/. 

This repository contains the code, and results/figures included in the paper.


### Project Structure
The project is structured as shown in the flowchart. 

* Dataset collection/creation:
The different datasets are collected (and created) using the data_import.py and 
ImageNet_subset_creator.py files in the io-folder. Specific paths to the different datasets are to be set in the
data_paths.py file. The PCam-dataset is first converted to PNG-files and stored in the home directory using the
pcam_converter.py file. The PNG-images are collected using the data_import.py file. 

* Pretraining:
In the transfer_experiments.py file the pretraining experiment is created and connected with Sacred. In the models 
folder the model_preparation_saving.py and tf_generators_models_kfold.py files include functions that create the 
necessary model, generators, etc. After pretraining the trained model is stored on OSF using the requests_OSF.py file. 
The experiment results are logged into Neptune using Sacred. 

* Transfer learning and evaluation:
The pretrained models are used in the transfer learning experiments, created in the transfer_experiments.py file. 
Similarly to pretraining, models, generators etc. are created using the model_preparation_saving.py and 
tf_generators_models_kfold.py files. The transfer performance is evaluated using the AUC_evaluation.py file in the
evaluation folder. The resulting models, weights and predictions are stored on OSF with the 
requests_OSF.py file. The experiment results are logged into Neptune using Sacred. 
Figures included in the paper are created using the visualization functions in the AUC_evaluation.py file. Note that 
for this the trained models need to be in the home directory.

Extra: 
Feature maps of the models can be created using the featuremaps_viz.py file, plots showing the stability during 
training/validation/testing can be created using the stability_plot.py file. 




<img src="figures/Flowchart_CatScans_subproject1.png" alt="flowchart">

### Prerequisites

The packages needed to run the project are listed in the requirements.txt file.

The databases used in the project need to be downloaded and stored in the home directory of the project. Download links
can be found on the reference website of the dataset.

<!-- USAGE EXAMPLES -->
## Usage

In the top section of the transfer_experiments.py file an experiment can be defined. An example:

First give the experiment a meaningful name.
```shell script
ex = Experiment('Resnet_pretrained=isic_target=pcam-middle')
```
Define arguments for the experiment such as parameter settings, which datasets to use, etc.
```shell script
target = True
source_data = "isic"
target_data = "pcam-middle"
x_col = "path"
y_col = "class"
augment = True
k = 5
img_length = 96
img_width = 96
learning_rate = 0.000001
batch_size = 128
epochs = 20
color = True
dropout = 0.5
scheduler_bool = False
home = '/data/'
```
Run the experiment by running the python file. Include server specifications if necessary.
```shell script
python -m src.transfer_experiments.py 
```
