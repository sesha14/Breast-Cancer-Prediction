# Breast-Cancer-Prediction
Using the provided dataset, implement 4 different classification algorithms to determine if a candidate has breast cancer. 
Data:
The attached csv file contains all the data. The run file handles importing it and converting it to numpy arrays. A description of the dataset is in the run file.
K-Nearest Neighbor:
Use Euclidean distance between the features. Choose a k value and use majority voting to determine the class. The k value is provided to the knn class. Please implement train (which is just memorizing the data) and predict methods (that run the knn algorithm). The distance function is provided for you and you can assume all data is continuous. In case of a tie, you can pick either class.
Decision Tree:
Use the ID3 algorithm as defined in the slides. Since all the data is continuous they have to be broken down into bins. This has been done for you in the preprocess method. You can preprocess the data in the train and predict methods (that you implement) before evaluating. The preprocess functions breaks the data into bins based on equal width. If there are ties in selecting the majority class or the maximum information gain, pick one.
Perceptron:
For the perceptron, multiply the inputs by a weight matrix and then pass the output through a single heaviside (step function) function to get the output. Don’t forget the bias. Train the perceptron using the perceptron learning algorithm. The weight matrix and bias are initialized in the run file to facilitate grading. You must update the weights, but don’t change the shape or type of the numpy array. The number of steps are defined in the run file, for each step update the model on a single datapoint.
Multi-Layer perceptron (MLP):
The number of steps are defined in the run file, for each step update the model on a single datapoint.
