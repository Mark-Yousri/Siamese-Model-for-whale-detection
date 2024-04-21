# Siamese Model For Whale Detection

This is a project that aims to train a siamese network for image similarity using whale images. A siamese network is a type of neural network that learns to compare two inputs and output a similarity score. In this project, we use the Xception model as the base encoder to extract features from the images, and then use a custom distance layer to compute the Euclidean distance between the anchor, positive, and negative images. The goal is to minimize the distance between the anchor and the positive images, and maximize the distance between the anchor and the negative images.

## Dataset

The dataset used in this project is a subset of the Whale Recognition Challenge hosted by Kaggle. It contains 50 folders, each with 10 images of a different whale species. The images are resized to 128x128 pixels and named as 0.jpg, 1.jpg, ..., 9.jpg. The dataset is split into 90% train and 10% test, and then used to create triplets of images for the siamese network. A triplet consists of an anchor image, a positive image (same whale species as the anchor), and a negative image (different whale species from the anchor).

## Model

The model used in this project is a siamese network composed of three parts:

- Encode_Model: A sequential model that takes an image as input and outputs a 256-dimensional feature vector. It consists of a pretrained Xception model with average pooling, followed by a flatten layer, a dense layer with 512 units and ReLU activation, a batch normalization layer, a dense layer with 256 units and ReLU activation, and a lambda layer that normalizes the output vector.
- DistanceLayer: A custom layer that takes the feature vectors of the anchor, positive, and negative images as inputs and outputs the squared Euclidean distance between the anchor and the positive, and the anchor and the negative.
- Siamese_Network: A model that takes three images as inputs and outputs the two distances as outputs. It uses the Encode_Model and the DistanceLayer as sub-models.

The model is trained using the Adam optimizer with a learning rate of 1e-3 and an epsilon of 1e-01. The loss function is the triplet loss, which is defined as the maximum of zero and the sum of the margin (1.0) and the difference between the two distances. The model is evaluated using the accuracy metric, which is the proportion of triplets where the distance between the anchor and the positive is smaller than the distance between the anchor and the negative.

## Results

The model underwent training for a duration of **60 epochs** with a **batch size of 128**. Throughout the training process, there was a significant reduction in loss, plummeting from **0.21962** to a mere **0.00029**, while the test accuracy soared to an impressive **85.55%**. This performance indicates the model's proficiency in differentiating between various whale species through their visual representations, subsequently generating a similarity score that mirrors their likeness.

**Training Confusion Matrix**:
The confusion matrix from the training phase can be visualized below, demonstrating the model's ability to accurately classify the whale species.


![image](https://github.com/Mark-Yousri/Siamese-Model-for-whale-detection/assets/100801214/833ed576-eca9-4ac6-8957-e11a458eb282)










   
**Testing Confusion Matrix**:
The testing phase confusion matrix is available below, showcasing the model's effectiveness in generalizing its classification capabilities to unseen data.

![image](https://github.com/Mark-Yousri/Siamese-Model-for-whale-detection/assets/100801214/152b74c8-72b9-4440-b388-c47baaf4610a)

