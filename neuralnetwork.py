import numpy as np
import scipy.special


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.lr = learningrate
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndim=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_outputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        targets = np.array(targets_list, ndim=2).T
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who = self.who + self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                               np.transpose(hidden_outputs))
        self.wih = self.wih + self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                               np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndim=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activate_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_outputs)
        return final_outputs
