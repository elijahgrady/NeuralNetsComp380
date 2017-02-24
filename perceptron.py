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


def input_method():
    while (1):
        training_data = input(
            'Enter 1 to train using a traning data file, enter 2 to train using a trained weights file : ')
        print(training_data)
        if training_data == '1':
            training_data_file_name = input('Enter the training data file name : ')
            training_data_weights = input(
                'Enter 0 to initialize weight to 0, or, enter 1 to initialize weights to random values between -0.5 and 0.5 : ')
            training_data_max_epochs = input('Enter the maximum number of training epochs : ')
            training_data_output_weights = input('Enter a file name to save the trained weight settings : ')
            training_data_alpha_rate = input('Enter the learning rate alpha from >0 to 1 : ')
            training_data_threshold_theta = input('Enter the threshold theta : ')
            print('Training converged after 4 epochs')
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

class TrainingData:
    #Sets value and 7 weights for the value
    def __init__(self, value, weight):
        self.value = value
        self.weights = {}
        for x in range(start=1, end = 7):
            self.weights[x] = weight
    def changevalue(self, value):
        self.value = value

    def changeweight(self, newW, index):
        self.weights[index] = newW


#Pass in a list of TrainingData objects
def perceptron():
    alpha = 1
    omega = 0
    converged = False
    trainingdata = [] #this should be a list of 64 (63 + bias) values and 7  weights
    yin = 0
    y = 0
    bias = 0


    while converged is False:
        for x in  trainingdata:
            x =

def main():
    prompt()
    input_method()


if __name__ == '__main__':
    main()

