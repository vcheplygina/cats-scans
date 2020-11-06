## Subproject 2 - Dataset similarities

### Project abstract
The aim of this project is to investigate the relation between the performance of a transfer learned model and the similarity of the source- and target datasets. In order to do this, medical as well as non-medical datasets were analyzed by means of different types of meta-features, experts- and statistical meta-features respectively. Combining experts’ input and statistical measures has, as known so far, not  been done yet and has the goal of getting a broader knowledge of the relation between dataset characteristics and model performance. It could also expose the potential of human judgement in (source) dataset selection as opposed to statistical measures. The experts meta-features are based on answers to a questionnaire about the datasets filled in by experts, whereas the latter are based on statistical measures of the image histograms. From these results, the Euclidean distance is determined for different combinations of source- and target datasets. These distances, which are a measure of the similarity between two datasets, are then compared to the AUC score of the model. The results show no correlation between similarity, based on either one of the two types of meta-features, and the performance of the model.


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)



<!-- ABOUT THE PROJECT -->
## About The Project
5 datasets were used for the experiments. These are the following:
1. [Chest X-rays](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
3. [ISIC2018 - Task 3 - the training set](https://challenge2018.isic-archive.com/task3/training/)
4. [STL-10](https://cs.stanford.edu/~acoates/stl10/)
5. [PatchCamelyon (PCam)](http://basveeling.nl/posts/pcam/)

Considering the initial size of the PCAM dataset (327.680 images), a subset of 100.000 images was used to speed up training and calculation times.

### Project Structure
The project is structured as shown in the flowchart. 

* Dataset import (collection/creation) subsets/folder etc.:
..

* Experts similarity matrix:
..

* Satistical similarity matrix:
..

<img src="Flowchart_CatScans_subproject2.png" alt="flowchart">

### Prerequisites

The packages needed to run the project are listed in the requirements.txt file.

All the datasets used in this project are located in the local_data folder which can be downloaded via this link: LINK

<!-- USAGE EXAMPLES -->
## Usage

All the experiments can be executed from the similarity_experiment.py file. An example:

First specify the absolute path to where the downloaded local_data folder is located.
```shell script
absolute_path_local_data = 'C:/example_path/local_data'
```
Define arguments for the experiment such as parameter settings, which datasets to use, etc.
```shell script
datasets_list = ['chest_xray', 'dtd', 'ISIC2018', 'stl-10', 'pcam']
defined_subset = 'None'
```
Run the experiment by running the appropriate section. Include server specifications if necessary.
```shell script
expert_mfe = norm_and_invert(expert_answers(expert_answer_path=absolute_path_local_data))
```


<!-- ROADMAP -->
## Roadmap

See [project](https://github.com/vcheplygina/cats-scans/projects/1) for a list of issues that are used to create the 
project.



