# Train CV 3

This is a project that aims to train a siamese network for image similarity using whale images. A siamese network is a type of neural network that learns to compare two inputs and output a similarity score. In this project, we use the Xception model as the base encoder to extract features from the images, and then use a custom distance layer to compute the Euclidean distance between the anchor, positive, and negative images. The goal is to minimize the distance between the anchor and the positive images, and maximize the distance between the anchor and the negative images.

## Dataset

The dataset used in this project is a subset of the [Whale Recognition Challenge](^1^) hosted by Kaggle. It contains 50 folders, each with 10 images of a different whale species. The images are resized to 128x128 pixels and named as 0.jpg, 1.jpg, ..., 9.jpg. The dataset is split into 90% train and 10% test, and then used to create triplets of images for the siamese network. A triplet consists of an anchor image, a positive image (same whale species as the anchor), and a negative image (different whale species from the anchor).

## Model

The model used in this project is a siamese network composed of three parts:

- Encode_Model: A sequential model that takes an image as input and outputs a 256-dimensional feature vector. It consists of a pretrained Xception model with average pooling, followed by a flatten layer, a dense layer with 512 units and ReLU activation, a batch normalization layer, a dense layer with 256 units and ReLU activation, and a lambda layer that normalizes the output vector.
- DistanceLayer: A custom layer that takes the feature vectors of the anchor, positive, and negative images as inputs and outputs the squared Euclidean distance between the anchor and the positive, and the anchor and the negative.
- Siamese_Network: A model that takes three images as inputs and outputs the two distances as outputs. It uses the Encode_Model and the DistanceLayer as sub-models.

The model is trained using the Adam optimizer with a learning rate of 1e-3 and an epsilon of 1e-01. The loss function is the triplet loss, which is defined as the maximum of zero and the sum of the margin (1.0) and the difference between the two distances. The model is evaluated using the accuracy metric, which is the proportion of triplets where the distance between the anchor and the positive is smaller than the distance between the anchor and the negative.

## Results

The model is trained for 60 epochs with a batch size of 128. The training loss decreases from 0.21962 to 0.00029, and the test accuracy reaches 1.00000. The model is able to learn to distinguish different whale species based on their images, and output a similarity score that reflects their resemblance. The model is saved as a pickle file named siamese_model.pkl.
