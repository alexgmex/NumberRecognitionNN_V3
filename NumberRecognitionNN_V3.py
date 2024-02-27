import pandas as pd
import numpy as np

# Neural network class
class Network:

    def __init__(self, layers):
        
        # Initialise empty arrays, adding 1 extra layer to the neurons for the input layer
        self.neurons = [np.zeros(layers[0])]
        self.weights = []
        self.biases = []

        # For layers, add neurons, weights and biases in range (-1,1)
        for i in range(1, len(layers)):
            self.neurons.append(np.zeros(layers[i]))
            self.weights.append(2*np.random.rand(layers[i],layers[i-1])-1)
            self.biases.append(2*np.random.rand(layers[i])-1)
    



    def forward_propagate(self, input_layer):

        # Set first layer of neurons to be the input
        self.neurons[0] = input_layer

        # Forward propagate
        for i in range(len(self.neurons)-1):
            self.neurons[i+1] = ReLU(np.dot(self.weights[i], self.neurons[i]) + self.biases[i])

        # Return guess and output layer
        return np.argmax(self.neurons[-1]), self.neurons[-1]
    



    def back_propagate(self, answer):

        # Assign learning rate
        learning_rate = 0.05

        # One-hot encoding of answer key
        key = np.zeros(10)
        key[answer] = 1

        # Create output error
        output_error = (key - self.neurons[-1]) * delta_ReLU(self.neurons[-1])

        # Reshape output error to be (10,1) and last hidden layer to be (1,16)
        transposed_error = output_error.reshape((output_error.shape[0], 1))
        transposed_final_layer = self.neurons[-2].reshape(1,self.neurons[-2].shape[0])

        # Perform matrix multiplication and update weights
        weight_delta = learning_rate * np.matmul(transposed_error, transposed_final_layer)
        self.weights[-1] = self.weights[-1] + weight_delta

        # Update biases
        bias_delta = learning_rate * output_error
        self.biases[-1] = self.biases[-1] + bias_delta




# CONVERT CSV DATA INTO AN ARRAY
def csv_to_arr(filepath):
    arr = pd.read_csv(filepath,header=None).to_numpy()
    answers = arr[:,0]
    data = arr[:,1:]
    return answers, data




# RELU FUNCTIONS
def ReLU(value):
    return np.where(value >= 0, value, 0)

def delta_ReLU(value):
    return np.where(value >= 0, 1, 0)



def main():

    # Get training and testing data
    train_answers, train_data = csv_to_arr("MNIST_Data/mnist_train.csv")
    test_answers, test_data = csv_to_arr("MNIST_Data/mnist_test.csv")

    # Establish network layer counts
    NN = Network([784, 16, 16, 10])

    # Train network
    for x in range(2): #len(train_answers)
        guess, output_layer = NN.forward_propagate(train_data[x]/255)
        print(f"Guess: {guess}. Answer: {train_answers[x]}")
        NN.back_propagate(train_answers[x])


if __name__ == "__main__":
    main()