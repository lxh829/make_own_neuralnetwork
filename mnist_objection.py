import numpy as np
import scipy.special

####①定义了一个神经网络
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        targets = np.array(targets_list, nd=2).T
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who = self.who + self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                               np.transpose(hidden_outputs))
        self.wih = self.wih + self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                               np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
#####②将神经网络实例化并且输入初始的参数
inputnodes = 784
hiddennodes = 200
outputnodes = 10
learningrate = 0.1

n = neuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)

####③导入训练数据从csv文件
training_data_file = open('mnist_train_100.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
print(training_data_list)

###④通过进行多个世纪来进行计算
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(outputnodes) + 0.01
        targets[int(all_values[0])] = 0.99
        pass
###⑤导入测试文件
test_data_file = open('mnist_test_10.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

###⑥测试，给一个图片得到预测值
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
####⑦计算出准确率
scorecard_array = np.asarray(scorecard)
print(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)