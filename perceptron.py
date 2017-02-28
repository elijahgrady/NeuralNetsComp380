# implement a computer program to classify letters from different fonts using Perceptron
# learning (see page 74 of the text)
# the program uses the input and output data dimensions specified in its network training
# set (see attached training set on page 72 for more detail)
# however the program can be applied to pattern classification problems with any data dimensions
# (as long as the dimensions od its training and testing sets are consistent)
# test your program through the appropriate dimension-matched testing sets
# (see attached sample on page 75 for detail)
# save resulting testing files into the following format

"""
Actual Output:
A
1 -1 -1 -1 -1 -1 -1
Classified Output:
A
1 -1 -1 -1 -1 -1 -1
"""

# The following experiments were performed and reports have been included:

'''
1.
Train your net by a traning dataset (example attached sample training data) then use the same set as a testing set.
Does the net classify the training samples correcetly?

2.
For a fixed testing set, test your net by selecting several values for the learning rate alpha (.25-->1.00)
Select several values for threshold theta(0.00, 0.25, .50, 1.00, 5.00, 10.00, 50.00)
Present your results in a table and draw your conclusions...

3.
After training your net with the attached traning set, test the ability of the net
(in terms of its classification accuracy, or percentage of correctly classified letters) to noisy versions of the traning patterns.
Create three testing sets in this experiment:
    LNITest --> LNI input patterns (p75)
    MNITest -->
    HNITest
Set alpha = 0.25 but try different values of the threshold theta
Present your results in a table
Draw your conclusion if there is one
'''


storage = []
vector = []
output = []
def prompt():
    print('Welcome to our first neural network - A Perceptron Net!\n')
    print('\n')


def quit_method():
    training_data_test_deploy = input('Enter 1 to test/deploy using a testing/deploying ddata file, enter 2 to quit : ')
    if training_data_test_deploy == '2':
        print('Thanks for playing with the Perceptron Net! Goodbye!')
        exit(0)
        return '0'
    else:
        return '1'


# this class is an dictionary of size xy (63 values) and TargetNum target values for the output
class TrainingData:
    # there's a smarter way to implement this
    # where we pass in an iterable of pos indexes and iterable of pos targets and set those to 1
    # but i wrote it this way already so we can fix if needed
    def __init__(self, x, y, TargetNum):
        self.values = {}
        self.targets = {}
        #from 1 to 63
        for i in range(1, (x*y)+1):
            self.values[i] = -1
        for i in range(1, TargetNum +1):
            self.targets[i] = -1

    def setindex(self, indexes, value):
        for x in indexes:
            self.values[x] = value

    def settargets(self, indexes, value):
        for x in indexes:
            self.targets[x] = value

#this class has a value and a set of associated weights
class Neuron:
    def __init__(self, value, weight, numWeights):
        self.value = value
        self.weights = {}
        for x in range(1,numWeights +1):
            self.weights[x] = weight

    def changeValue(self, value):
        self.value = value

    def changeWeight(self, newW, index):
        self.weights[index] = newW

#This class will have neurons x1,x2,x3...xnumNeurons and a bias neuron, with specified weight
#should default value be something other than 0? Possibly -1?
#I don't think this is an issue, because the neuron value is assigned by training data\
class Net:
    def __init__(self, numNeurons, weight, numWeights):
        self.neurons = {}
        for x in range(1, numNeurons):
            temp = Neuron(0,weight, numWeights)
            self.neurons[x] = temp
        temp = Neuron(0,weight, numWeights)
        self.neurons['bias'] = temp

def initializeStuff(s):
    global inputDimension
    global outputDimension
    global numberOfTraining
    global stringVector
    global vectors
    global storage
    global output
    vectors = []
    f = open(s,'r')
    f.readline()
    f.readline()
    inputDimension = [int(s) for s in f.readline().split() if s.isdigit()]
    outputDimension = [int(s) for s in f.readline().split() if s.isdigit()]
    numberOfTraining =  [int(s) for s in f.readline().split() if s.isdigit()]


    #Read all the training data set
    for i in range(0,numberOfTraining[0]):
        f.readline()
        m = f.readline()
        x =len(m.strip())

        kk = [True for i in m if i.isalpha()]
        while (x !=0  and kk != True):
            storage.append(m.strip("\n"))
            m = f.readline()
            x = len(m.strip().replace(" ", ""))
            kk = [True for i in m if i.isalpha()]
        

        stringVector = ' '.join(x.strip() for x in storage if x.strip())
        # print("vector is %s" % stringVector)
        vectors.insert(i,stringVector)
        print("List is Vector[%s]=%s\n" % (i, vectors[i]))
        # outputstring = f.readline().strip("\n").replace(" ", "")
        output.insert(i,f.readline().strip("\n").replace(" ", ""))
        print(output)

        letter = f.readline()
        print("Letter is %s" % letter)
        storage = []


    #initialize the training data     






  





def main():
    global weight
    prompt()
    while (1):
        training_data = input(
            'Enter 1 to train using a training data file, enter 2 to train using a trained weights file : ')
        if training_data == '1':
            training_data_file_name = input('Enter the training data file name : ')
            #Parse stuff for the file

            training_data_weights = input(
                'Enter 0 to initialize weight to 0, or, enter 1 to initialize weights to random values between -0.5 and 0.5 : ')
            if (training_data_weights == 1):
                weight = random.uniform(-0.5, 0.5)
            else:
                weight = 0


            initializeStuff(training_data_file_name)


            training_data_max_epochs = input('Enter the maximum number of training epochs : ')
            training_data_output_weights = input('Enter a file name to save the trained weight settings : ')
            training_data_alpha_rate = input('Enter the learning rate alpha from >0 to 1 : ')
            training_data_threshold_theta = input('Enter the threshold theta : ')
            #Under what conditon it should print this?
            print('Training converged after %s epochs' % training_data_max_epochs)
            if (quit_method()) == '1':
                training_data_deploy_filename = input('Enter the testing/deploying data file name : ')
                training_data_deploy_results = input('Enter a file name to save the testing/deploying results : ')
                print('\n')
                print('[Training through trained weight files]')
                prompt()
                continue
            if (quit_method()) == '2':
                break
        if training_data == '2':
            trained_weight_settings = input('Enter the trained weight setting input data file name : ')
            if (quit_method()) == '2':
                break
            if (quit_method()) == '1':
                '''
                same as above but for option 2
                will implement this once option 1 is complete
                should be a lot faster that way
                '''

    #TODO fix prompt messages so they pass back needed variables
    #TODO figure out format of training data, convert that data into TrainingData objects

    # these are our net variables, will need to be passed from those prompt and input methods


    for font in vectors:
        for xx in font.split():
            # TrainingData(int(x),weight,numberOfTraining[0])
            x = int(xx)
            y = weights
            dimensions = x * y
            outputClasses = 7
            weight = weight
            alpha = training_data_alpha_rate
            threshold = training_data_threshold_theta


            converged = False #boolean if our learning has converged
            change = False #boolean if weights have been changed

            yin = {} #this is yin in the book equations
            for x in range(1, outputClasses+1):
                yin[x] = 0

            yf = {} #this is 'y' in the book equations
            for x in range(1,outputClasses +1):
                yf[x] = 0


            myNet = Net(dimensions, weight, outputClasses)
            #myNet.neurons[INDEX].value is how to reference xi
            #myNet.neurons[INDEX].weights[wINDEX] is how to reference wij

            A = TrainingData(x,y, outputClasses)
            Aindexes = [3,4,11,18,24,26,31,33,37,38,39,40,41,44,48,51,55,57,58,59,61,62,63]
            Atargets = [1]
            A.settargets(Atargets,1)
            A.setindex(Aindexes, 1)

            # This print statement will correctly print a dictionary with all specified indexes 1 and non specified -1
            #print(A.values)
            #print(A.targets)

            #List of our training samples, as TrainingData objects
            trainingSamples = []
            trainingSamples.append(A)


            #PERCEPTRON
            while(converged is False):

                for x in trainingSamples:

                    for y in range(1,dimensions +1):
                        myNet.neurons[y].value = x.values[y] #this should say xi = si, this runs from x1 to x63

                    for j in range(1, outputClasses + 1): #from 1 to 7

                        for z in range(1, dimensions +1): #from 1 to 63, generate yin[j]

                            yin[j]= yin[j] + (myNet.neurons[z].value * myNet.neurons[z].weights[j]) #yin[j] = x1w1j + x2w2j + ...

                        yin[j] = yin[j] + myNet.neurons['bias'].weights[j] #yin[j] also needs wb[j] added

                        #yf[j] = f(yin[j])
                        if yin[j] < threshold:
                            yf[j] = -1
                        elif yin[j] > threshold:
                            yf[j] = 1
                        else:
                            yf[j] = 0

                    # I don't know if this needs to be in a different loop than the one above it,
                    # but it is
                    # this just checks if y is different than the target, if it is, it updates the weights

                    for j in range(1, outputClasses +1): # j runs 1 - 7
                        if yf[j] != x.targets[j]:
                            change = True
                            for i in range(1, dimensions +1): # i runs 1 - 63
                                myNet.neurons[i].weights[j] = myNet.neurons[i].weights[j] + (alpha * x.targets[j] * myNet.neurons[i].value)
                                #should say wij(new) = wij(old) + (alpha tj xi)
                            myNet.neurons['bias'].weights[j] = myNet.neurons['bias'].weights[j] + (alpha * x.targets[j])
                            #this should say wbj(new) = wbj(old) + (alpha tj)

                #if we did not change anything, then our learning converged
                if change is False:
                    converged = True
    print(yf)



if __name__ == '__main__':
    main()

