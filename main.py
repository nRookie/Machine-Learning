import numpy
import matplotlib.pyplot

# scipy.special for the sigmoid function expit()
import scipy.special

# neural network class definition
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate
        

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc

        # self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))


        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)


        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer 
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        # hidden laer error is the output_errors, split by weights, recombined at hidden nodes.
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the link between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    #learning_rate is 0.3
    learning_rate = 0.3

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    output = n.query([1.0, 0.5, -1.5])
    print(output)

    data_file = open("mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    all_values = data_list[0].split(',')
    scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    print(scaled_input)
    image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.savefig("samplePlot2.png")
    data_file.close()
    print(len(data_list))



def main1():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list

    training_data_file = open("mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network

    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record 
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    pass

    test_data_file = open("mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    all_values = test_data_list[0].split(',')
    print(all_values[0])

    image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap = 'Greys', interpolation = 'None')
    matplotlib.pyplot.savefig("samplePlot3.png")
    print (n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))

def whole_set():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list

    training_data_file = open("mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network

    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record 
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    pass
    scorecard =[]

    test_data_file = open("mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    print(len(test_data_list))

    for record in test_data_list:
        all_values = record.split(',')

        correct_label = int(all_values[0])
        print(correct_label, "correct label")

        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        print(label, "network's answer")

        # append correct or incorrect to list 
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

    print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)
    print ("performance percentage = ", (scorecard_array.sum() / scorecard_array.size)*100,"%")



def whole_set_big():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    # create instance of neural network
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data CSV file into a list

    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network

    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record 
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    pass
    scorecard =[]

    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    print(len(test_data_list))

    for record in test_data_list:
        all_values = record.split(',')

        correct_label = int(all_values[0])
        # print(correct_label, "correct label")

        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # print(label, "network's answer")

        # append correct or incorrect to list 
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

    print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)
    print ("performance percentage = ", (scorecard_array.sum() / scorecard_array.size)*100,"%")


whole_set_big()