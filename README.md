![](icons/resized/project%20logo.png)

### background

Project goal: Detection of plant species and diseases from images.
We conducted extensive research on various architectures to optimize our results. Initially, we developed a custom CNN network. Initially, we engineered a custom CNN network, achieving high test accuracy with clean data. However, encountering challenges posed by noisy data prompted us to pivot towards pre-trained models. Leveraging transfer learning with established models like VGG, AlexNet, DenseNet, and ResNet, we applied methodologies such as fine-tuning, feature extraction, DoRA, and LoRA. Ultimately, the adoption of the vision transformers architecture emerged as the most effective solution in addressing noise sensitivity. Throughout each phase, we trained the models and assessed their performance across validation and test sets.

# Plant Village dataset
We utilized a dataset comprising 38 classes encompassing both diseased and healthy plants. This dataset was sourced from the following link: [PlantVillage](//https://paperswithcode.com/dataset/plantvillage "PlantVillage").
example image: 

![](images/plant1.png)

![](images/plant2.png)
