# Cats or CAT Scans 



# Background

Networks for medical image classification can be pretrained on a variety of datasets, from natural images such as cats, to other medical image datasets such as CT scans. There is no consensus on what strategy is best, an overview of several papers which investigate this is here: https://arxiv.org/abs/1810.05444 . 
Two factors seem to be important:

1) Source data needs to be "large enough" 
2) Source data needs to be "similar" to target data

The exact definitions of these factors are not given, nor is their relative importance. This is what we want to investigate in this project, with these two questions as subprojects.  

# Preliminary results

Some preliminary results are available in the thesis of Floris Fok (code https://github.com/tueimage/transfer-medical-bsc-2018, report available as PDF):
* Source datasets used are ImageNet (1M images), CatDog (25K), Natural (8K), Chest (6K), DiabeticRetinopathy (35K) 
* Target datasets used are Chest (6K), ISIC 2017 (2K), BloodCell (12.5K)
* Averaged over different target datasets, ImageNet is the best source, and DiabeticRetinopathy is the second best 
* Transfer learning with fine-tuning outperforms transfer learning by feature extraction 


# Projects

## Baseline / minimum viable solution / ...  
Both projects rely on having several datasets, trained networks, and results on target datasets. To avoid duplication, the first mini-project is to reproduce (part of) the preliminary results:

* Define a list of datasets you might want to use throughout the whole project
* From this list, select two source datasets and one target
* Train two different networks on both sources (each of you trains a different type of network) 
* Test networks on the target
* Compare results 

This will create some initial data, to start developing further experiments. You should write your own code in this repository (both projects in the same repository), but you can use other code as needed. 


## Subproject 1 - Dataset properties 
* Select several source datasets 
* Train networks on systematically varied versions of the datasets, where a property (such as size) is varied
* Save the datasets / trained networks for reuse
* Test on target data to investigate how these properties influence performance


## Subproject 2 - Dataset similarity
* Define multiple ways to measure similarity
    * High-level properties (such as medical or not) 
    * Feature vector representation of the images
    * Trained network weights
    * Asking an observer
    * Etc
* Measure similarity of datasets 
    * If available, use an already saved dataset/trained network
    * If not, train/test the network and save it
* Compare similarity matrices to performances to investigate which similarity is more suitable




# Resources

* https://github.com/tueimage/transfer-medical-bsc-2018 - preliminary results
* https://github.com/tueimage/transfer-medical-msc-2019 - transfer learning experiments on two datasets, varying properties, could be useful for code structure and general transfer learning mechanisms 
* https://github.com/tueimage/meta-segmentation-msc-2018/ - meta-learning to represent a dataset as a feature vector, could be useful for similarity measures 

* https://github.com/tueimage/essential-skills - tutorial about deep learning, Github etc. 
* https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004947 - article about Github

* https://modelzoo.co/ - pretrained networks
* https://github.com/IDSIA/sacred - toolbox for keeping track of experiments (optional) 

# Papers

* References in "Cats or CAT scans" paper
* Transfusion: Understanding Transfer Learning with Applications to Medical Imaging http://papers.nips.cc/paper/8596-transfusion-understanding-transfer-learning-for-medical-imaging (+ maybe citations of this paper)
* What And How Other Datasets Can Be Leveraged For Medical Imaging Classification https://ieeexplore.ieee.org/document/8759148


# Project management

We will use Github Issues, and the board in the Projects tab, to manage the project. Any coding-related to-do's (for example, implementing a function) should be created as an issue, because this allows adding comments, labels, deadlines etc. Then from the project board, you can import this issue as a card. 

A good to-do should be actionable (start with a verb, such as implement or read) and concrete (for example, a specific functionality, a number of papers, etc). 

Cards start in the "backlog" column, and then progress through "to-do" (plan for next week), "in progress" (doing it now) and "done". The idea is to have as few things as possible in the "in progress" column, and to prioritize what you have in "to-do". This is based on the agile or Kanban methdology. You might want to search for "examples Kanban research projects" or similar queries to get an idea of how people use it for such projects. 




