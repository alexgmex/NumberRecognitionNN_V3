import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Neural network class
class Network:

    def __init__(self, layers):
        
        # Initialise empty arrays, adding 1 extra layer to the neurons for the input layer
        self.neurons = [np.zeros(layers[0])]
        self.weights = []
        self.biases = []
        self.n_error = []
        self.w_error = []
        self.b_error = []
        
        

        # For layers, add neurons, weights and biases in range (0,1)
        for i in range(1, len(layers)):
            self.neurons.append(np.zeros(layers[i]))
            self.weights.append(2*np.random.rand(layers[i],layers[i-1])-1)
            self.biases.append(2*np.random.rand(layers[i])-1)
            self.n_error.append(np.zeros(layers[i]))
            self.w_error.append(np.zeros((layers[i],layers[i-1])))
            self.b_error.append(np.zeros(layers[i]))
            
    



    def forward_propagate(self, input_layer):

        # Set first layer of neurons to be the input
        self.neurons[0] = input_layer

        # Forward propagate until last layer
        for i in range(len(self.weights)-1):
            self.neurons[i+1] = activation(np.dot(self.weights[i], self.neurons[i]) + self.biases[i])

        # Softmax last layer
        i_final = len(self.weights) - 1
        self.neurons[i_final + 1] = softmax(np.dot(self.weights[i_final], self.neurons[i_final]) + self.biases[i_final])

        # Return guess and output layer
        return np.argmax(self.neurons[-1]), self.neurons[-1]
    



    def back_propagate(self, answer):

        # Assign learning rate
        learning_rate = 0.01

        # One-hot encoding of answer key
        key = np.zeros(10)
        key[answer] = 1

        # Calculate output error (WIP)
        self.n_error[-1] = (key - self.neurons[-1])

        # Loop through hidden layers and calculate errors
        for i in range(-1, -len(self.weights), -1):

            # Reshape then tile previous layer sum
            error_sum = self.n_error[i].reshape((self.n_error[i].shape[0], 1))
            error_sum = np.tile(error_sum, self.weights[i].shape[1])
            
            # Multiply and sum them with the weights
            error_sum = error_sum * self.weights[i]
            error_sum = np.sum(error_sum, 0, keepdims=True)

            # Flatten then multiply with derivative of neuron values
            error_sum = error_sum.flatten()
            self.n_error[i-1] = delta_activation(self.neurons[i-1]) * error_sum

        
        # Update weights and biases
        for i in range(len(self.weights)):
            # Reshape output error to be (10,1) and last hidden layer to be (1,16)
            transposed_error = self.n_error[i].reshape((self.n_error[i].shape[0], 1))
            transposed_hidden = self.neurons[i].reshape((1,self.neurons[i].shape[0]))

            # Perform matrix multiplication
            self.w_error[i] = learning_rate * np.matmul(transposed_error, transposed_hidden)
            self.b_error[i] = learning_rate * self.n_error[i]

            # Update weights and biases
            self.weights[i] = self.weights[i] + self.w_error[i]
            self.biases[i] = self.biases[i] + self.b_error[i]



# CONVERT CSV DATA INTO AN ARRAY
def csv_to_arr(filepath):
    arr = pd.read_csv(filepath,header=None).to_numpy()
    answers = arr[:,0]
    data = arr[:,1:]
    return answers, data




# TRANSFER FUNCTIONS
def activation(value):
    return np.where(value >= 0, value, 0.1*value) # Leaky ReLU
    # return (np.exp(value) - np.exp(-value)) / (np.exp(value) + np.exp(-value)) # Tanh
    # return 1.0 / (1.0 + np.exp(-value)) # Sigmoid

def delta_activation(value): 
    return np.where(value >= 0, 1, 0.1) # Leaky ReLU
    # return 1.0 - np.power(activation(value),2) # Tanh
    # return activation(value)*(1.0 - activation(value)) # Sigmoid

def softmax(value):
    return np.exp(value)/np.sum(np.exp(value))




def main():

    # Get training and testing data
    train_answers, train_data = csv_to_arr("MNIST_Data/mnist_train.csv")
    test_answers, test_data = csv_to_arr("MNIST_Data/mnist_test.csv")
    bias_answers, bias_data = csv_to_arr("MNIST_Data/mnist_bias.csv")


    # Make accuracy counters
    train_arr = np.zeros(len(train_answers))
    test_counter = 0


    # Establish network layer counts
    NN = Network([784, 100, 100, 10])


    # Train network
    for x in range(len(train_answers)): 
        guess, output_layer = NN.forward_propagate(train_data[x]/255)
        NN.back_propagate(train_answers[x])
        print(f"Training... {round(100*x/len(train_answers))}%")
        if guess == train_answers[x]: train_arr[x] = 1
        
    
    # Test network
    for x in range(len(test_answers)): 
        guess, output_layer = NN.forward_propagate(test_data[x]/255)
        print(f"Guess: {guess}. Answer: {test_answers[x]}")
        if guess == test_answers[x]: test_counter += 1
    

    # Print final accuracy and create rolling average of results

    print(f"\nFinal accuracy: {round(100*test_counter/len(test_answers),2)}%")

    rolling_avg = []
    x_axis = []

    for x in range(499, len(train_answers)):
        rolling_avg.append(100*np.sum(train_arr[x-500:x])/500)
        x_axis.append(x)
    
    plt.plot(x_axis, rolling_avg)
    plt.title("Rolling accuracy of training over time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()


if __name__ == "__main__":
    main()