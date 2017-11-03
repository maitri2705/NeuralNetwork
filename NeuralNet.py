import numpy as np
import sys
import pandas as pd

def linearActivation(x, factor = False):
    if factor:
        return 1.0
    else:
        return x

def sigmoidActivation(x,factor = False):
    if factor:
        return (sigmoidActivation(x) * (1-sigmoidActivation(x)))
    else:
        return (1/(1+np.exp(-1*x)))


class NeuralNetwork:
    layerCount = 0
    shape = None
    err=0
    weights = []
    b = []

    ''' ========================  INITIALIZE PARAMETERS ============================='''
    def initialize_parameters(self, layer_dimensions,activationFunctions = None):
        self.layer_dimensions=layer_dimensions
        self.layerCount = len(layer_dimensions) - 1 # dont need to count input layer as official layer

        self.shape = layer_dimensions

        self.input = []
        self.output = []
        self.lastWeightDelta = []

        if activationFunctions is None:
            functions = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1: # means output layer
                    functions.append(linearActivation)
                else:
                    functions.append(sigmoidActivation)

        else:
                functions = activationFunctions[1:]

        self.tempActivationFunction = functions


        for(i,j) in zip(layer_dimensions[:-1],layer_dimensions[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(j,i+1))) #for transposing weights like W^T and  i+1 is for adding ones at last column
            self.lastWeightDelta.append(np.zeros([j,i+1]))


    ''' ========================  FORWARD PASS =============================='''
    def forwardPass(self,input):

        inputInstances = input.shape[0]


        self.inputArray = []
        self.outputArray = []

        bias = np.ones([1,inputInstances])

        for i in range(self.layerCount):
            if i == 0:
                input = self.weights[0].dot(np.vstack([input.T, bias]))
            else:
                input = self.weights[i].dot(np.vstack([self.outputArray[-1],bias]))

            self.inputArray.append(input)

            self.outputArray.append(self.tempActivationFunction[i](input))


        return self.outputArray[-1].T  # -1 because we need last output layer's output

    ''' ========================  BACKWARD PASS ============================='''
    def backwardPass(self, input, target, trainingRate = 0.001, momentumAlpha = 0.9):
        deltaArray = []
        inputInstacne = input.shape[0]

        self.forwardPass(input)

        for i in reversed(range(self.layerCount)):
            if i == self.layerCount - 1:
                you_got = self.outputArray[i]
                difference = you_got - target.T

                #------mean squared error----------
                error = np.mean(difference**2)

                x = self.inputArray[i]

                deltaArray.append(difference * self.tempActivationFunction[i](x, True))
            else:
                deltaHidden = self.weights[i+1].T.dot(deltaArray[-1]) # sigma operation

                deltaArray.append(deltaHidden[:-1, :] * self.tempActivationFunction[i](self.inputArray[i], True))

        # for weight deltas
        for i in range(self.layerCount):
            indexOfDelta = self.layerCount - 1 - i

            if i == 0:
                output = np.vstack([input.T,np.ones([1,inputInstacne])])
            else:
                output = np.vstack([self.outputArray[i-1], np.ones([1,self.outputArray[i-1].shape[1]])])

            currentWeightDelta = output.dot(deltaArray[indexOfDelta].T).T
            # Add momentum term
            weightDelta = (trainingRate * currentWeightDelta) + (momentumAlpha * self.lastWeightDelta[i])

            self.weights[i] = self.weights[i] - (trainingRate * weightDelta)
            self.lastWeightDelta[i] = weightDelta

        return error

    ''' ======================== START WEIGHT PRINTING ============================='''
    def printNodeWeight(self):
        n=self.layerCount
        for i in range(n):
            if(i==0):
                print("Layer ",i,"(input layer):")
            elif(i==n):
                print("Layer ",i,"(last hidden layer):")
            else:
                print("Layer ",i,"(hidden layer number:",i,"):")


            temp=(self.weights[i].T[:-1])

            for j in range(self.layer_dimensions[i]):
                print("     Neuron",j," weights:",temp[j])
        print("Training Error:",self.err)

    ''' ========================  END WEIGHT PRINTING ============================='''

    ''' ========================  START TRAIN THE NETWORK ============================='''
    def trainTheNetwork(self, input, target, maxIteration, ourErr):
        print("=====================Neural network training has been initialized==========================\n")
        print("Processing...")
        for i in range(maxIteration + 1):
            err = self.backwardPass(input, target)
            self.err=err

            if err <= ourErr:
                print("Minimum error reached at Iteration {0}".format(i), "error: {0}".format(err))
                break

        # .........after weights are modified we need to run network one more time..........
        self.forwardPass(input)
        print("=====================Neural network has been trained==========================\n")
    ''' ======================== END OF TRAIN THE NETWORK ============================='''

    '''=================== START of test test data================================'''
    def testNetwork(self,input,target):
        self.forwardPass(input)

        for i in reversed(range(self.layerCount)):
            if i == self.layerCount - 1:
                you_got = self.outputArray[i]
                difference = you_got - target
                error=np.mean(difference**2)
        print("Testing error:",error)
    '''=================== END of test test data================================'''

''' ========================  MAIN FUNCTION ============================='''
if __name__ == "__main__":
    nn = NeuralNetwork()
    if (sys.argv.__len__()) > 1:
        inputFilePath = sys.argv[1]
    else:
        print("Please enter Filepath of data")

    ### read file####
    df=pd.read_csv(inputFilePath,header=None)

    ### row drop###
    df=df.drop(df.index[0])

    col=df.columns.tolist()
    df=pd.read_csv(inputFilePath,usecols=col[1:len(col)],header=None)

    df=df.drop(df.index[0])

    row_count=df.shape[0]
    splitingFactor = int(sys.argv[2])/100
    train_df=df.loc[0:(int(row_count*splitingFactor))]

    test_df=df.loc[(int(row_count*splitingFactor))+1:]

    train_target = train_df.iloc[:,-1]
    test_target = test_df.iloc[:,-1]
    train=train_df.drop(train_df.columns[-1],axis=1)
    test=test_df.drop(test_df.columns[-1],axis=1)

    number_hidden=int(sys.argv[4])

    initial_parameter=[]
    initial_parameter.append(len(train.T))

    for i in range (number_hidden):
        initial_parameter.append(int(sys.argv[i+5]))

    initial_parameter.append(1)

    nn.initialize_parameters(initial_parameter)

    maxIteration=int(sys.argv[3])
    ourErr=1e-5

    nn.trainTheNetwork(train.as_matrix(),train_target.as_matrix(),maxIteration,ourErr)
    nn.printNodeWeight()
    nn.testNetwork(test.as_matrix(),test_target.as_matrix())
'''============END OF MAIN================'''