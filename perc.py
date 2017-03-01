import random

# implement a computer program to classify letters from different fonts using Perceptron
# learning (see page 74 of the text)
# the program uses the input and output data dimensions specified in its network training
# set (see attached training set on page 72 for more detail)
# however the program can be applied to pattern classification problems with any data dimensions
# (as long as the dimensions od its training and testing sets are consistent)
# test your program through the appropriate dimension-matched testing sets
# (see attached sample on page 75 for detail)
# save resulting testing files into the following format

storage = []
vector = []
output = []

outputFile = ''
output_classifications_file = ''
output_classifications_list = list()
letter = list()


def prompt():
    print('Welcome to our first neural network - A Perceptron Net!\n')
    print('\n')


def quit_method():
    training_data_test_deploy = input('Enter 1 to test/deploy using a testing/deploying data file, enter 2 to quit : ')
    if training_data_test_deploy != '1':
        print('Thanks for playing with the Perceptron Net! Goodbye!')
        exit(0)
    else:
        return '1'


# this class is an dictionary of size xy (63 values) and TargetNum target values for the output
class TrainingData:
    def __init__(self, values, dimensions, TargetNum, output):
        self.values = {}
        self.targets = {}
        self.yf = {}
        # from 1 to 63
        count = 0
        for x in values.split(' '):
            try:
                x = int(x)
                count = count + 1
                self.values[count] = x
            except:
                pass
        count = 0
        for i in output.split(' '):
            try:
                i = int(i)
                count = count + 1
                self.targets[count] = i
            except:
                pass
        for i in range(1, TargetNum + 1):
            self.yf[i] = 0


# this class has a value and a set of associated weights
class Neuron:
    def __init__(self, value, weight, numWeights, option):
        self.value = value
        self.weights = {}
        if option:
            count = 0
            for value in weight.split():
                count += 1
                self.weights[count] = float(value)
        else:
            for x in range(1, numWeights + 1):
                self.weights[x] = weight


# This class will have neurons x1,x2,x3...xnumNeurons and a bias neuron, with specified weight
# should default value be something other than 0? Possibly -1?
# I don't think this is an issue, because the neuron value is assigned by training data\
class Net:
    def __init__(self, numNeurons, weight, numWeights, option):
        self.neurons = {}

        for x in range(1, numNeurons + 1):
            if option:
                temp = Neuron(0, weight[int(x - 1)], numWeights, option)
            else:
                temp = Neuron(0, weight, numWeights, option)
            self.neurons[x] = temp
        if not option:
            temp = Neuron(0, weight, numWeights, option)
        if option:
            temp = Neuron(0, weight[int(63)], numWeights, option)
        self.neurons['bias'] = temp


class InitVars:
    def __init__(self, inputDimension, outputDimension, numTrain, data, output):
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.numTrain = numTrain
        self.data = data
        self.output = output


def parseWeight(weights):
    global storage
    weightsFile = open(weights, 'r')
    weightcontainer = []
    # parse the weight
    for i in range(0, 64):
        m = weightsFile.readline()
        storage.append(m.strip("\n"))
        stringVector = ' '.join(x.strip() for x in storage if x.strip())
        weightcontainer.insert(i, stringVector)
        storage = []
    return weightcontainer


def initializeStuff(s, weight):
    global storage
    output = []
    vectors = []
    f = open(s, 'r')
    f.readline()  # nothing
    f.readline()  # sample testing set
    inputDimension = [int(s) for s in f.readline().split() if s.isdigit()]
    outputDimension = [int(s) for s in f.readline().split() if s.isdigit()]
    numberOfTraining = [int(s) for s in f.readline().split() if s.isdigit()]

    # Read all the training data set
    for i in range(0, numberOfTraining[0]):
        f.readline()
        m = f.readline()
        x = len(m.strip())

        kk = [True for i in m if i.isalpha()]
        while (x != 0 and kk != True):
            storage.append(m.strip("\n"))
            m = f.readline()
            x = len(m.strip().replace(" ", ""))
            kk = [True for i in m if i.isalpha()]

        stringVector = ' '.join(x.strip() for x in storage if x.strip())

        vectors.insert(i, stringVector)

        output.insert(i, f.readline().strip("\n"))

        letter.append(f.readline())

        storage = []

    # initialize the training data
    data = []
    # data should be from
    count = 0
    for x in vectors:
        data.append(TrainingData(x, inputDimension[0], outputDimension[0], output[count]))
        count = count + 1

    return InitVars(inputDimension[0], outputDimension[0], numberOfTraining[0], data, output)


def main():
    global weight
    global outputFile
    global output_classifications_file
    global output_classifications_list
    global letter
    prompt()
    while (1):
        training_data = input(
            'Enter 1 to train using a training data file, enter 2 to train using a trained weights file : ')
        training_data = check_input_1(training_data)
        if training_data == '1':
            training_data_file_name = input('Enter the training data file name : ')
            training_data_file_name = check_input_2(training_data_file_name)
            training_data_weights = input(
                'Enter 0 to initialize weight to 0, or, enter 1 to initialize weights to random values between -0.5 and 0.5 : ')
            training_data_weights = check_input_3(training_data_weights)
            if (training_data_weights == 1):
                weight = float(random.uniform(-0.5, 0.5))
            else:
                weight = 0
            myvars = initializeStuff(training_data_file_name, weight)
            training_data_max_epochs = input('Enter the maximum number of training epochs : ')
            training_data_max_epochs = check_input_4(training_data_max_epochs)
            training_data_output_weights = input('Enter a file name to save the trained weight settings : ')
            training_data_output_weights = check_input_5(training_data_output_weights)
            outputFile = training_data_output_weights
            print("outputfile for weights name is %s" % outputFile)
            training_data_alpha_rate = input('Enter the learning rate alpha from >0 to 1 : ')
            training_data_alpha_rate = check_input_6(training_data_alpha_rate)
            training_data_threshold_theta = input('Enter the threshold theta : ')
            training_data_threshold_theta = check_input_7(training_data_threshold_theta)
            output_classifications_file = input('Enter a file name to save the testing/deploying results : ')
            output_classifications_file = check_input_8(output_classifications_file)
            print("Training the perceptron...")
            perceptron(myvars.inputDimension, myvars.outputDimension, myvars.data,
                       weight, training_data_alpha_rate, training_data_threshold_theta, training_data_max_epochs, False,
                       output_classifications_file)
            if (quit_method()) == '2':
                break
            else:
                training_data_deploy_filename = input('Enter the testing/deploying data file name : ')
                training_data_deploy_filename = check_input_9(training_data_deploy_filename)
            myvars = initializeStuff(training_data_deploy_filename, weight)
            print('Testing the perceptron...')
            perceptron(myvars.inputDimension, myvars.outputDimension, myvars.data, weight, training_data_alpha_rate,
                       training_data_threshold_theta, training_data_max_epochs, False, output_classifications_file)
            print('\n')
            prompt()
            continue
        if training_data == '2':
            training_data_weight_file_name = input('Enter the trained weight setting input data file name : ')
            training_data_weight_file_name = check_input_10(training_data_weight_file_name)
            if (quit_method()) == '1':
                training_data_deploy_filename = input('Enter the testing/deploying data file name : ')
                training_data_deploy_filename = check_input_9(training_data_deploy_filename)
                myvars = initializeStuff(training_data_deploy_filename, None)
                print('Testing the perceptron...')
                outputFile = input('Enter a file name to save the testing/deploying results : ')
                outputFile = check_input_8(outputFile)
                perceptron(myvars.inputDimension, myvars.outputDimension, myvars.data,
                           parseWeight(training_data_weight_file_name), 1, 0, 5, True, outputFile)
                print('\n')
                print('[Training through trained weight files]')
                prompt()


def perceptron(inputD, outputD, data, weight, alpha, threshold, maxepochs, option, file):
    if not option:
        m = open(outputFile, 'a+')

    # these are our net variables
    dimensions = inputD
    outputClasses = outputD

    converged = False  # boolean flag if our learning has converged

    yin = {}  # this is yin in the book equations
    for x in range(1, outputClasses + 1):
        yin[x] = 0

    myNet = Net(dimensions, weight, outputClasses, option)  # Net construction
    for x in range(1, dimensions + 1):
        for y in range(1, outputClasses + 1):
            print(myNet.neurons[x].weights[y])
        for y in range(1, outputClasses + 1):
            print(myNet.neurons['bias'].weights[y])

    # List of our training samples, as TrainingData objects
    trainingSamples = data

    # PERCEPTRON
    epochs = 0
    while (converged is False):
        count = 0
        epochs = epochs + 1
        change = 0

        for x in trainingSamples:
            # set yin to 0
            for g in range(1, outputClasses + 1):
                yin[g] = 0
            # Check if max epochs
            if epochs >= int(maxepochs):
                print("Training converged after", epochs, "epochs, the maximum amount.")
                converged = True
                break
            # add to training sample counter
            count = count + 1

            for y in range(1, dimensions + 1):
                myNet.neurons[y].value = x.values[y]  # this should say xi = si, this runs from x1 to x63

            for j in range(1, outputClasses + 1):  # from 1 to 7

                for z in range(1, dimensions + 1):  # from 1 to 63, generate yin[j]
                    yin[j] = yin[j] + (
                        myNet.neurons[z].value * myNet.neurons[z].weights[j])  # yin[j] = x1w1j + x2w2j + ...

                yin[j] = yin[j] + myNet.neurons['bias'].weights[j]  # yin[j] also needs wb[j] added

                # yf[j] = f(yin[j])
                if yin[j] < float(threshold):
                    x.yf[j] = -1
                elif yin[j] > float(threshold):
                    x.yf[j] = 1
                else:
                    x.yf[j] = 0

            for j in range(1, outputClasses + 1):

                if (x.yf[j] - x.targets[j]) > .001 or (x.targets[j] - x.yf[j]) > .001:
                    change = change + 1
                    for i in range(1, dimensions + 1):  # i runs 1 - 63
                        myNet.neurons[i].weights[j] = myNet.neurons[i].weights[j] + (
                        float(alpha) * x.targets[j] * myNet.neurons[i].value)
                        # should say wij(new) = wij(old) + (alpha tj xi)
                    myNet.neurons['bias'].weights[j] = myNet.neurons['bias'].weights[j] + (float(alpha) * x.targets[j])
                    # this should say wbj(new) = wbj(old) + (alpha tj)

        # if we did not change anything, then our learning converged
        if change is 0:
            print("Converged after", epochs, "epochs.")
            converged = True
            break

    f = open(file, 'a+')
    count = 0
    for x in trainingSamples:
        # Actual Output
        # A
        #
        f.write("Actual Output:\n")
        l = list(letter[count])
        f.write(str(l[0]))
        f.write("\n")
        count = count + 1

        for j in range(1, outputClasses + 1):
            f.write(str(x.targets[j]))
        f.write("\nClassified Output:\n")
        classletter = ""
        lettercount = 0
        if x.yf[1] is 1:
            classletter = "A"
            lettercount = lettercount + 1
        elif x.yf[2] is 1:
            classletter = "B"
            lettercount = lettercount + 1
        elif x.yf[3] is 1:
            classletter = "C"
            lettercount = lettercount + 1
        elif x.yf[4] is 1:
            classletter = "D"
            lettercount = lettercount + 1
        elif x.yf[5] is 1:
            classletter = "E"
            lettercount = lettercount + 1
        elif x.yf[6] is 1:
            classletter = "J"
            lettercount = lettercount + 1
        elif x.yf[7] is 1:
            classletter = "K"
            lettercount = lettercount + 1

        if lettercount is 1:
            f.write(str(classletter))
            f.write("\n")
        else:
            f.write("Indeterminate letter\n")
        for j in range(1, outputClasses + 1):
            f.write(str(x.yf[j]))
        f.write("\n\n")
    f.close()

    # write the weights to the output file
    format = 0
    if not option:
        for x in range(1, dimensions + 1):
            for j in range(1, outputClasses + 1):
                m.write(str(myNet.neurons[x].weights[j]) + ' ')
                format += 1
                if (format % 7 == 0):
                    m.write('\r\n')
        for j in range(1, outputClasses + 1):
            m.write((str(myNet.neurons['bias'].weights[j])) + ' ')
        m.close()


def check_input_1(training_data):
    count = 0
    while (1):
        if training_data is '1':
            return training_data
        if training_data is '2':
            return training_data
        else:
            count += 1
            if count > 2:
                print('too many strikes, you are outta here!')
                exit(0)
            training_data = input(
                'Enter 1 to train using a training data file, enter 2 to train using a trained weights file : ')


def check_input_2(training_data_file_name):
    suffix = '.txt'
    count = 0
    while (1):
        if training_data_file_name.endswith(suffix):
            return training_data_file_name
        else:
            count += 1
            if count > 2:
                print('too many strikes, you are outta here!')
                exit(0)
            training_data_file_name = input('Enter the training data file name : ')


def check_input_3(training_data_weights):
    count = 0
    while (1):
        if training_data_weights is '1':
            return training_data_weights
        if training_data_weights is '0':
            return training_data_weights
        else:
            count += 1
            if count > 2:
                print('too many strikes, you are outta here!')
                exit(0)
            training_data_weights = input(
                'Enter 0 to initialize weight to 0, or, enter 1 to initialize weights to random values between -0.5 and 0.5 : ')


def check_input_4(training_data_max_epochs):
    count = 0
    while (1):
        if count > 1:
            print('too many strikes, you are outta here')
            exit(0)
        if training_data_max_epochs.isdigit() != True:
            count += 1
            training_data_max_epochs = input('Enter the maximum number of training epochs : ')
        if training_data_max_epochs.isdigit():
            if int(training_data_max_epochs) > 0:
                return training_data_max_epochs
            else:
                count += 1
                training_data_max_epochs = input('Enter the maximum number of training epochs : ')


def check_input_5(training_data_output_weights):
    suffix = '.txt'
    count = 0
    while (1):
        if training_data_output_weights.endswith(suffix):
            return training_data_output_weights
        else:
            if count > 2:
                print('too many strikes, you are outta here!')
                exit(0)
            training_data_output_weights = input('Enter a file name to save the trained weight settings : ')


def check_input_6(training_data_alpha_rate):
    count = 0
    while (1):
        try:
            training_data_alpha_rate = float(training_data_alpha_rate)
        except:
            print('sorry, no strikes for completely invalid input! you are outta here!')
            exit(0)
        if count > 2:
            print('too many strikes, you are outta here')
            exit(0)
        if float(training_data_alpha_rate) <= 1.000:
            if float(training_data_alpha_rate) > 0:
                return training_data_alpha_rate
        count += 1
        training_data_alpha_rate = input('Enter the learning rate alpha from >0 to 1 : ')


def check_input_7(training_data_threshold_theta):
    count = 0
    while (1):
        if count > 1:
            print('too many strikes, you are outta here')
            exit(0)
        if training_data_threshold_theta.isdigit() != True:
            count += 1
            training_data_threshold_theta = input('Enter the threshold theta : ')
        if training_data_threshold_theta.isdigit():
            if int(training_data_threshold_theta) > 0:
                return training_data_threshold_theta
            else:
                count += 1
                training_data_threshold_theta = input('Enter the threshold theta : ')


def check_input_8(output_classifications_file):
    suffix = '.txt'
    count = 0
    while (1):
        if output_classifications_file.endswith(suffix):
            return output_classifications_file
        else:
            if count > 2:
                print('too many strikes, you are outta here!')
                exit(0)
            output_classifications_file = input('Enter a file name to save the testing/deploying results : ')


def check_input_9(training_data_deploy_filename):
    suffix = '.txt'
    count = 0
    while (1):
        if training_data_deploy_filename.endswith(suffix):
            return training_data_deploy_filename
        else:
            if count > 2:
                print('too many strikes, you are outta here!')
                exit(0)
            training_data_deploy_filename = input('Enter the testing/deploying data file name : ')


def check_input_10(training_data_weight_file_name):
    suffix = '.txt'
    count = 0
    while (1):
        if training_data_weight_file_name.endswith(suffix):
            return training_data_weight_file_name
        else:
            if count > 2:
                print('too many strikes, you are outta here!')
                exit(0)
            training_data_weight_file_name = input('Enter the trained weight setting input data file name : ')


if __name__ == '__main__':
    main()
