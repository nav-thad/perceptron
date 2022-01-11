# Navalan Thadchanamoorthy
# November 2021 - January 2022
# Objective: designing object-oriented neural network that takes user inputted number of folds, epochs, and learning rate
#inspiration: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

#imports
from random import seed
from random import randrange
from csv import reader

# Data Elements:
#            - self.dataset: the dataset that is being subjected to binary classification
#            - self.nFolds: the number of folds being made in the dataset 
#            - self.learnRate: the rate of learning (controls magnitude to which weights can be updated)
#            - self.nEpochs: the number of epochs (times the data is iterated while updating the weights)
# Methods:
#            - __init__: constructs the object of the class
#            - prepData: converts columns of data from strings to float values, then to integers to get all values between 0-1
#            - cross_validation_split and evaluate_algorithm: estimates skill of machine learning model 
#                   - randomizes dataset, then splits into k groups; uses 1 group as test data set and rest as training data
#                   - fits model to training set then evaluates it on test data set, retains evaluation score and discards model
#            - predict: passes product of weight and input through activation function
#            - train_weights: estimates the weight values using stochastic gradient descent
#            - perceptron: carries out the algorithm of a perceptron (single layer neural network)
# Functions:
#            - loadFile: extracts data from csv file


class NeuralNetwork:
    def __init__(self, dataset, nFolds, learnRate, nEpochs):
        self.dataset = dataset
        self.nFolds = nFolds
        self.learnRate = learnRate
        self.nEpochs = nEpochs
        self.prepData()

    def prepData(self):
        for i in range(len(self.dataset[0]) - 1):
            for row in self.dataset:
                row[i] = float(row[i].strip())

        values = [row[len(self.dataset[0]) - 1] for row in self.dataset]
        tempSet = set(values)
        tempDict = dict()
        for i, value in enumerate(tempSet):
            tempDict[value] = i
        for row in self.dataset:
            row[len(self.dataset[0])-1] = tempDict[row[len(self.dataset[0])-1]]

    def cross_validation_split(self): 
        split = list()
        copy = list(self.dataset)
        fold_size = int(len(self.dataset) / self.nFolds)
        for i in range(self.nFolds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(copy))
                fold.append(copy.pop(index))
            split.append(fold)
        return split

    def evaluate_algorithm(self, *args): 
        folds = self.cross_validation_split()
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            test = list()
            for row in fold:
                row_copy = list(row)
                test.append(row_copy)
                row_copy[-1] = None
            predicted = self.perceptron(trainSet, test, *args)
            actual = [row[-1] for row in fold]

            accuracy = 0
            for i in range(len(actual)):
                if actual[i] == predicted[i]:
                    accuracy += 1
            accuracy = accuracy / float(len(actual)) * 100.0 
            scores.append(accuracy)
        
        return scores

    def predict(self, row, weights):
        returnVal = 0.0
        activation = weights[0]
        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]
        if activation >= 0.0:
            returnVal = 1.0
        return returnVal
    
    def train_weights(self, train):
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(self.nEpochs):
            for row in train:
                prediction = self.predict(row, weights)
                error = row[-1] - prediction
                weights[0] = weights[0] + self.learnRate * error
                for i in range(len(row)-1):
                    weights[i + 1] = weights[i + 1] + self.learnRate * error * row[i]
        return weights
    
    def perceptron(self, train, test):
        predictions = list()
        weights = self.train_weights(train)
        for row in test:
            prediction = self.predict(row, weights)
            predictions.append(prediction)
        return(predictions)

# Main

def loadFile(name):
	dataset = list()
	with open(name, 'r') as file:
		fileReader = reader(file)
		for row in fileReader:
			if row == "\n":
				continue
			dataset.append(row)
	return dataset

seed(1)

# User Inputs 

folds = int(input("Enter the number of folds: "))
learnRate = float(input("Enter the learning rate: "))
epochs = int(input("Enter the number of epochs: "))
datasetName = str(input("Enter the dataset file name (include .csv): "))
network = NeuralNetwork(loadFile(datasetName), folds, learnRate, epochs)
print("For the dataset titled %s: " % datasetName)
network = NeuralNetwork(loadFile(datasetName), folds, learnRate, epochs)
scores = network.evaluate_algorithm()
print('Scores:       %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
