from Layer import Layer
from Training import Training
import Activations


class NeuralNet:
    def __init__(self, inputNeuronCount: int, trainingBatchSize: int, outputNeuronCount: int):
        self.layers = []
        if len(self.layers) != 0:
            raise Exception("Neural net already has an input layer")

        self.trainingBatchSize = trainingBatchSize
        self.set = Training(trainingBatchSize, inputNeuronCount, outputNeuronCount)

        inputLayer = Layer(inputNeuronCount)
        inputLayer.SetInputLayerActivationMatrix(inputNeuronCount, trainingBatchSize)
        self.layers.append(inputLayer)

    def AddLayer(self, neuronCount: int):
        if len(self.layers) < 1:
            raise Exception("Neural net requires an input layer before adding hidden layers")

        self.layers[-1].SetActivationFunction(Activations.ReLU, Activations.ReLUDerivative)

        hiddenLayer: Layer = Layer(neuronCount)
        hiddenLayer.SetPreviousLayer(self.layers[-1], self.trainingBatchSize)

        self.layers.append(hiddenLayer)

        self.layers[-1].SetActivationFunction(Activations.LogisticFunction, Activations.LogisticFunctionDerivative)

    def FeedForward(self):
        if len(self.layers) <= 1:
            raise Exception("Net cannot be fed forward with the current layer layout")

        self.set.GetNextBatch()

        inputBatch: Matrix = self.set.GetInputBatch()
        self.layers[0].CopyInputLayerActivationMatrix(inputBatch)
        self.layers[0].FeedForward()

    def FeedBack(self):
        if len(self.layers) <= 1:
            raise Exception("Net cannot be fed backward with the current layer layout")

        targets: Matrix = self.set.GetOutputBatch()
        self.layers[-1].FeedBack(targets, self.trainingBatchSize)

    def PrintInputLayer(self):
        self.layers[0].PrintActivations()

    def PrintOutputLayer(self):
        self.layers[-1].PrintActivations()
