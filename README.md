# CIFAR-10 Image Classification using CNN from scratch and transfer learning with 87% accuracy

This repository contains code for performing image classification on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) implemented in a Jupyter Notebook. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

This project aims to demonstrate the process of building and training a CNN model to classify images into their respective classes.

## Notebook Contents

- Dataset Loading: The notebook begins by loading the CIFAR-10 dataset using the `tf.keras.datasets.cifar10` module. The dataset is divided into training and testing sets.

- Data Preprocessing: The images in the dataset are preprocessed by normalizing the pixel values and performing one-hot encoding on the class labels.

- Model Architecture: The CNN model architecture is defined using the `tf.keras.Sequential` API. It consists of multiple convolutional layers, max-pooling, and fully connected layers. The model summary is displayed, showcasing the number of parameters and the layer configuration.

- Model Training: The model is trained using the training set images and labels. The training process involves specifying the loss function, optimizer, and metrics for evaluation. The training progress, including the loss and accuracy, is logged and displayed.

- Model Evaluation: The trained model is evaluated on the testing set to measure its performance. The accuracy of the model on the testing set is displayed.

- Prediction Example: An example is provided to showcase how to use the trained model to predict the class labels of unseen images.

- Transfer Learning: Model building using Transfer Learning with DenseNet121 architecture

## Requirements

To run the code in this notebook, you need the following dependencies:

- Python (3.6 or higher)
- TensorFlow (2.0 or higher)
- Jupyter Notebook

You can install the required packages by running the following command:

```shell
pip install tensorflow jupyter
```
## CNN from scratch architecture
![image](https://github.com/DiptiSanap/CIFAR10-image-classification-CNN/assets/107847530/f4f6b965-f545-422b-abdb-6a6ca6d07857)



## Results of CNN from scratch model

The accuracy of the trained CNN model on the CIFAR-10 testing set is 87%. The detailed results, including the loss and accuracy curves, can be found in the notebook.
![cifar10_30_epochs](https://github.com/DiptiSanap/CIFAR10-image-classification-CNN/assets/107847530/45e274c0-cf03-47c9-9502-3f09862298d5)

## Results of transfer learning using DenseNet121 architecture
![image](https://github.com/DiptiSanap/CIFAR10-image-classification-CNN/assets/107847530/03edf7c3-0616-4666-9d72-df2663f51bb2)




## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

Please feel free to modify the README file as per your requirements and add any additional details or explanations about the project.
