from __future__ import annotations
from Matrix import Matrix


class Layer:
    def __init__(self, neuronCount: int):
        self.neuronCount = neuronCount
        self.nextLayer = False
        self.previousLayer = False
        self.learning_rate = 0.5

    def SetInputLayerActivationMatrix(self, neuronCount: int, trainingBatchSize: int):
        self.activations = Matrix(neuronCount, trainingBatchSize)

    def CopyInputLayerActivationMatrix(self, matrix: Matrix):
        self.activations.COPY(matrix)

    def SetPreviousLayer(self, previousLayer: Layer, trainingBatchSize: int):
        self.previousLayer = previousLayer
        self.previousLayer.nextLayer = self

        self.activations = Matrix(self.neuronCount, trainingBatchSize)
        self.deltaActivations = Matrix(self.neuronCount, trainingBatchSize)

        self.sums = Matrix(self.neuronCount, trainingBatchSize)
        self.deltaSums = Matrix(self.neuronCount, trainingBatchSize)

        self.weights = Matrix(self.neuronCount, self.previousLayer.neuronCount)
        self.weights.Randomize()
        self.deltaWeights = Matrix(self.neuronCount, self.previousLayer.neuronCount)
        self.deltaWeightsTmp = Matrix(self.neuronCount, self.previousLayer.neuronCount)

        self.biases = Matrix(self.neuronCount, 1)
        self.biases.Randomize()
        self.deltaBiases = Matrix(self.neuronCount, trainingBatchSize)

    def FeedForward(self):
        if self.previousLayer:
            self.sums.mat_MUL(self.weights, self.previousLayer.activations)
            self.sums.AddColumnVector(self.biases)
            self.sums.ApplyFunction(self.activations, self.activationFunction)

        if self.nextLayer:
            self.nextLayer.FeedForward()

    def FeedBack(self, targets: Matrix, trainingBatchSize: int):
        if self.previousLayer:
            self.deltaBiases.COPY(self.activations)
            self.sums.ApplyFunction(self.deltaSums, self.derivative)

            if not self.nextLayer:
                self.deltaBiases.SUB(targets)
            else:
                self.deltaWeightsTmp.Transpose(self.nextLayer.weights)
                self.deltaBiases.mat_MUL(self.deltaWeightsTmp, self.nextLayer.deltaBiases)

            self.deltaBiases.Hadamard(self.deltaSums)
            self.deltaActivations.Transpose(self.previousLayer.activations)
            self.deltaWeights.mat_MUL(self.deltaBiases, self.deltaActivations)

            averageDeltaBias = Matrix(self.neuronCount, 1)
            averageDeltaWeight = Matrix(self.neuronCount, 1)

            self.deltaBiases.GetAverageColumnVector(averageDeltaBias)
            self.deltaWeights.GetAverageColumnVector(averageDeltaWeight)

            averageDeltaBias.scalar_MUL(self.learning_rate / trainingBatchSize)
            averageDeltaWeight.scalar_MUL(self.learning_rate / trainingBatchSize)

            self.biases.SubColumnVector(averageDeltaBias)
            self.weights.SubColumnVector(averageDeltaWeight)

            self.previousLayer.FeedBack(targets, trainingBatchSize)

    def SetActivationFunction(self, activationFunction, derivative):
        self.activationFunction = activationFunction
        self.derivative = derivative

    def PrintActivations(self):
        self.activations.PrintMatrix()
