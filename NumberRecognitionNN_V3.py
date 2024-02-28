import pandas as pd
import numpy as np

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
            self.biases.append(np.random.rand(layers[i]))
            self.n_error.append(np.zeros(layers[i]))
            self.w_error.append(np.zeros((layers[i],layers[i-1])))
            self.b_error.append(np.zeros(layers[i]))
            
    



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
        learning_rate = 0.005

        # One-hot encoding of answer key
        key = np.zeros(10)
        key[answer] = 1

        # Calculate output error
        self.n_error[-1] = (key - self.neurons[-1]) * delta_ReLU(self.neurons[-1])

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
            self.n_error[i-1] = delta_ReLU(self.neurons[i-1]) * error_sum

        
        # Update weights and biases
        for i in range(1, len(self.weights)):
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




# RELU FUNCTIONS
def ReLU(value):
    return np.where(value >= 0, value, 0.1*value)

def delta_ReLU(value):
    return np.where(value >= 0, 1, 0.1)



def main():

    # Make accuracy counter, get training and testing data
    accuracy_counter = 0
    train_answers, train_data = csv_to_arr("MNIST_Data/mnist_train.csv")
    test_answers, test_data = csv_to_arr("MNIST_Data/mnist_test.csv")
    bias_answers, bias_data = csv_to_arr("MNIST_Data/mnist_bias.csv")

    # Establish network layer counts
    NN = Network([784, 16, 16, 10])

    # Train network
    for x in range(len(train_answers)): 
        guess, output_layer = NN.forward_propagate(train_data[x]/255)
        NN.back_propagate(train_answers[x])
        print(f"Training... {round(100*x/len(train_answers))}%")
        # print(f"Guess: {guess}. Answer: {train_answers[x]}")

    # for x in range(len(bias_answers)): 
    #     guess, output_layer = NN.forward_propagate(bias_data[x]/255)
    #     NN.back_propagate(bias_answers[x])
    #     # print(f"Training... {round(100*x/len(train_answers))}%")
    #     print(f"Guess: {guess}. Answer: {train_answers[x]}")
    
    # Test network
    for x in range(len(test_answers)): 
        guess, output_layer = NN.forward_propagate(test_data[x]/255)
        print(f"Guess: {guess}. Answer: {test_answers[x]}")
        if guess == test_answers[x]: accuracy_counter += 1
    
    print(f"\nFinal accuracy: {round(100*accuracy_counter/len(test_answers))}%")

if __name__ == "__main__":
    main()