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
    if training_data_test_deploy == '2':
        print('Thanks for playing with the Perceptron Net! Goodbye!')
        exit(0)
        return '0'
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

    def setindex(self, indexes, value):
        for x in indexes:
            self.values[x] = value

    def settargets(self, indexes, value):
        for x in indexes:
            self.targets[x] = value


# this class has a value and a set of associated weights
class Neuron:
    def __init__(self, value, weight, numWeights, option):
        self.value = value
        self.weights = {}
        if option:
            count = 0
            for value in weight.split():
                count +=1
                print("aaaa %s" % value)
                self.weights[count] = value
        else:

            for x in range(1, numWeights + 1):
                self.weights[x] = weight

    def changeValue(self, value):
        self.value = value

    def changeWeight(self, newW, index):
        self.weights[index] = newW


# This class will have neurons x1,x2,x3...xnumNeurons and a bias neuron, with specified weight
# should default value be something other than 0? Possibly -1?
# I don't think this is an issue, because the neuron value is assigned by training data\
class Net:
    def __init__(self, numNeurons, weight, numWeights,option):
        self.neurons = {}

        for x in range(1, numNeurons + 1):

            temp = Neuron(0, weight[x], numWeights,option)
            self.neurons[x] = temp
        if not option:
            temp = Neuron(0, weight, numWeights,option)
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
    weightsFile = open(weights,'r')
    weightcontainer = []
    #parse the weight
    for i in range(0,64):
        m = weightsFile.readline()
        storage.append(m.strip("\n"))
        stringVector = ' '.join(x.strip() for x in storage if x.strip())
        # print("vector is %s" % stringVector)
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
        # print("storage is", storage, "\n")


        stringVector = ' '.join(x.strip() for x in storage if x.strip())
        # print("vector is %s" % stringVector)
        vectors.insert(i, stringVector)
        # print("List is Vector[%s]=%s\n" % (i, vectors[i]))
        # outputstring = f.readline().strip("\n").replace(" ", "")
        output.insert(i, f.readline().strip("\n"))
        # print("output is", output, "\n")

        letter.append(f.readline())
        # print("Letter is %s" % letter)
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
        if training_data == '1':
            training_data_file_name = input('Enter the training data file name : ')
            training_data_weights = input(
                'Enter 0 to initialize weight to 0, or, enter 1 to initialize weights to random values between -0.5 and 0.5 : ')
            if (training_data_weights == 1):
                weight = float(random.uniform(-0.5, 0.5))
            else:
                weight = 0
            myvars = initializeStuff(training_data_file_name, weight)
            training_data_max_epochs = input('Enter the maximum number of training epochs : ')
            training_data_output_weights = input('Enter a file name to save the trained weight settings : ')
            outputFile = training_data_output_weights
            print("outputfile for weights name is %s" % outputFile)
            training_data_alpha_rate = input('Enter the learning rate alpha from >0 to 1 : ')
            training_data_threshold_theta = input('Enter the threshold theta : ')
            print("Training the perceptron...")
            perceptron(myvars.inputDimension, myvars.outputDimension, myvars.data,
                       weight, training_data_alpha_rate, training_data_threshold_theta, training_data_max_epochs, False)
            if (quit_method()) == '2':
                break
            else:
                training_data_deploy_filename = input('Enter the testing/deploying data file name : ')
            myvars = initializeStuff(training_data_deploy_filename, weight)
            output_classifications_file = input('Enter a file name to save the testing/deploying results : ')
            print('Testing the perceptron...')
            perceptron(myvars.inputDimension, myvars.outputDimension, myvars.data, weight, training_data_alpha_rate, training_data_threshold_theta, training_data_max_epochs, False)
            print('\n')
            #call the output_classifications_method
            output_classifications_method(myvars.outputDimension, myvars.data, output_classifications_file)
            prompt()
            continue
        if training_data == '2':
            training_data_weight_file_name = input('Enter the trained weight setting input data file name : ')
            if (quit_method()) == '2':
                break
            if (quit_method()) == '1':
                training_data_deploy_filename = input('Enter the testing/deploying data file name : ')
                myvars = initializeStuff(training_data_deploy_filename, None)
                print('Testing the perceptron...')
                training_data_deploy_results = input('Enter a file name to save the testing/deploying results : ')
                outputFile = training_data_deploy_results
                perceptron(myvars.inputDimension, myvars.outputDimension, myvars.data, parseWeight(training_data_weight_file_name), 1, 1, 1, True)
                output_classifications_method(myvars.outputDimension, myvars.data, training_data_deploy_results)
                print('\n')
                print('[Training through trained weight files]')
                prompt()


def perceptron(inputD, outputD, data, weight, alpha, threshold, maxepochs, option):
    if not option:
        m = open(outputFile,'a+')

    # these are our net variables, will need to be passed from those prompt and input methods
    dimensions = inputD
    outputClasses = outputD

    converged = False  # boolean if our learning has converged


    yin = {}  # this is yin in the book equations
    for x in range(1, outputClasses + 1):
        yin[x] = 0
    

    myNet = Net(dimensions, weight, outputClasses, option)


    # myNet.neurons[INDEX].value is how to reference xi
    # myNet.neurons[INDEX].weights[wINDEX] is how to reference wij

    # List of our training samples, as TrainingData objects

    trainingSamples = data


    # PERCEPTRON
    epochs = 0
    while (converged is False):
        count = 0
        epochs = epochs + 1
        change = 0
        for x in trainingSamples:
            for g in range(1, outputClasses + 1):
                yin[g] = 0
            if epochs >= int(maxepochs):
                print("Training converged after", epochs, "epochs, the maximum amount.")
                converged = True
                break

            count = count + 1


            for y in range(1, dimensions + 1):
                myNet.neurons[y].value = x.values[y]  # this should say xi = si, this runs from x1 to x63

            for j in range(1, outputClasses + 1):  # from 1 to 7

                for z in range(1, dimensions + 1):  # from 1 to 63, generate yin[j]
                    yin[j] = yin[j] + (
                    myNet.neurons[z].value * myNet.neurons[z].weights[j])  # yin[j] = x1w1j + x2w2j + ...


                yin[j] = yin[j] + myNet.neurons['bias'].weights[j]  # yin[j] also needs wb[j] added

                # yf[j] = f(yin[j])
                if yin[j] < int(threshold):
                    x.yf[j] = -1
                elif yin[j] > int(threshold):
                    x.yf[j] = 1
                else:
                    x.yf[j] = 0

            for j in range(1, outputClasses +1):

                if (x.yf[j] - x.targets[j]) > .001 or (x.targets[j] - x.yf[j]) > .001:
                    change = change + 1
                    for i in range(1, dimensions + 1):  # i runs 1 - 63
                        myNet.neurons[i].weights[j] = myNet.neurons[i].weights[j] + (float(alpha) * x.targets[j] * myNet.neurons[i].value)
                        # should say wij(new) = wij(old) + (alpha tj xi)
                    myNet.neurons['bias'].weights[j] = myNet.neurons['bias'].weights[j] + (float(alpha) * x.targets[j])
                        # this should say wbj(new) = wbj(old) + (alpha tj)

                # if we did not change anything, then our learning converged

        if change is 0:
            #print("Converged after", epochs, "epochs.")
            #save our classifications
            for x in trainingSamples:
                for j in range(1, outputClasses+1):
                    output_classifications_list.append(x.yf[j])
            converged = True
            break

    #write the weights to the output file
    format = 0
        if not option:
        for x in range(1,dimensions +1):
            for j in range(1, outputClasses +1):
                m.write(str(myNet.neurons[x].weights[j]) + ' ')
                format +=1
                if (format%7==0):
                    m.write('\r\n')

            
        for j in range(1, outputClasses +1):
            m.write((str(myNet.neurons['bias'].weights[j])) + ' ')

        m.close()

def output_classifications_method(outputD2, trainingSamples2, s):
    n = open(s, 'a+')
    outputClasses = outputD2
    trainingSamples = trainingSamples2
    # write the actual and classified output to the correct outfile file
    # need to put this in a loop that goes over every object in our list of objects
    count = 0
    for x in trainingSamples:
        n.write('Actual Output:' + '\n')
        n.write(letter[count] + '\n')
        for j in range(1, outputClasses + 1):
            n.write(str(output_classifications_list[j]))
        n.write('\n')
        n.write('Classified Output:' + '\n')
        n.write(letter[count] + '\n')
        for j in range(1, outputClasses + 1):
            n.write(str(output_classifications_list[j]))
        n.write('\n')
        count += 1


if __name__ == '__main__':
    main()

