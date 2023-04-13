# Neural Network using Keras Library for Churn Modeling


Introduction

This project implements a neural network using the Keras library in Python to perform churn modeling. The dataset used for training and testing the model contains 14 columns and more than 1000 rows. The algorithm uses the ReLU activation function with two hidden layers and one input layer. The optimizer used is Adamax, and the loss function is binary crossentropy, which is commonly used for classification problems. The output layer contains a sigmoid function as the activation function.


Requirements

To run this program, you will need the following libraries installed:

Python 3.6 or higher

Keras 2.4.3 or higher

Pandas 1.0.5 or higher

NumPy 1.19.5 or higher

Matplotlib 3.2.2 or higher

Scikit-learn 0.23.1 or higher



Dataset

The churn modeling dataset used in this project contains 14 columns, including customer ID, surname, credit score, geography, gender, age, tenure, balance, number of products, credit card, active member, estimated salary, and whether the customer has churned or not. The dataset is divided into two parts: training and testing. The training set contains 80% of the data, and the testing set contains 20% of the data.



Model Architecture

The neural network consists of two hidden layers and one input layer. The input layer has 14 nodes, one for each feature in the dataset. The two hidden layers have 6 and 4 nodes, respectively. The output layer has 1 node, which predicts whether the customer will churn or not. The ReLU activation function is used for the hidden layers, and the sigmoid function is used for the output layer.



Training and Testing

The model is trained using the Adamax optimizer and binary crossentropy loss function. The batch size used is 10, and the number of epochs is 100. After training the model, it is evaluated on the testing set to determine its accuracy.



Results

The model achieved an accuracy of 86% on the testing set and a validation accuracy of 85%. The loss and validation loss are 0.3359 and 0.3547, respectively. This means that it correctly predicted whether a customer would churn or not 86% of the time. The confusion matrix and classification report are included in the program to provide more detailed information about the model's performance.


Conclusion

This program demonstrates how to use the Keras library in Python to implement a neural network for churn modeling. By using a neural network, we can achieve a high level of accuracy in predicting customer churn, which is crucial for businesses that want to retain their customers.




