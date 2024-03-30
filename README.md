![](icons/resized/project%20logo.png)


**Table of Contents**

[TOCM]

[TOC]

## Background
## Plant Village dataset
## Files in this repository
## Prerequisites
## How to run
## Results
## References

# background

Project goal: Detection of plant species and diseases from images.
We conducted extensive research on various architectures to optimize our results. Initially, we developed a custom CNN network. Initially, we engineered a custom CNN network, achieving high test accuracy with clean data. However, encountering challenges posed by noisy data prompted us to pivot towards pre-trained models. Leveraging transfer learning with established models like VGG, AlexNet, DenseNet, and ResNet, we applied methodologies such as fine-tuning, feature extraction, DoRA, and LoRA. Ultimately, the adoption of the vision transformers architecture emerged as the most effective solution in addressing noise sensitivity. Throughout each phase, we trained the models and assessed their performance across validation and test sets.

# Plant Village dataset
We utilized a dataset comprising 38 classes encompassing both diseased and healthy plants. This dataset was sourced from the following link: [PlantVillage](//https://paperswithcode.com/dataset/plantvillage "PlantVillage").
example image: 

![](images/plant1.png)  ![](images/plant2.png)

# Files in this repository
File Name   | Purpose
------------- | -------------
PlantDiseaseCnn.ipynb  | this file contains our custom CNN architecture, exploring the abilities and try different strategies in order to get the maximum accuracy. Utilizing Optuna and random search for hyperparameter optimization.
TransferLearningPartA.ipynb  | This notebook uses transfer learning to train existing models (AlexNet, VGG, ResNet, DenseNet). Its divided into two parts: Feature Extraction: Modifies only the last FC layer of each model for training. Fine-Tuning: Trains all layers of the model, starting from pretrained weights, focusing on AlexNet.
TransferLearningPartB.ipynb  | In this Python notebook, we applied transfer learning techniques using LoRA and DoRA. Each model underwent training and evaluation at every epoch using the validation set. Finally, we conducted a final assessment on the test set for each technique. You can find the accuracy and loss results of VGG16, AlexNet, ResNet, and DenseNet in this notebook.
PlantDiseaseVit.ipynb | This document outlines the preprocessing steps for our dataset tailored for the ViT architecture, along with the architecture itself. Additionally, it presents the outcomes of applying this architecture to both clean and noisy datasets.
utils/CNN_plot_utils.py
utils/ViT_utils.py
utils/project_results.py | This Python file contains functions to generate plots from CSV results. It accepts CSV files containing accuracy and training time data and plots the corresponding graphs.

# Prerequisites
 
| Library  | Version |
| ------------- | ------------- |
| python |   3.10.0 |
| pyTorch | |
| pandas |  |
| torchvision|  |
| numpy |  |
| tqdm  |   |

# How to run
Running Locally
Press "Download ZIP" under the green button Clone or download or use git to clone the repository using the following command: git clone https://github.com/LeeBenyamin/Sherleaf-Holmes-project.git (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

Open the folder in Jupyter Notebook (it is recommended to use Anaconda).

# Results

our cnn results:

Optuna VS Random search: 

![](results/optuna_VS_random_search.png) 

transfer learning results:

In this slide, we present the results obtained from our experiments, showcasing both graphical and tabular representations. The graph illustrates the relationship between accuracy test scores and training time for each model. Notably, the fine-tuned versions of AlexNet and ResNet demonstrate the highest accuracy among all models.

![](transfer_learning_results/graphs/all_net_plot.png) 

vison transformers results:

# References

