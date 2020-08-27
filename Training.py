from Matrix import Matrix


class Training:
    def __init__(self, trainingBatchSize: int, inputNeuronCount: int, outputNeuronCount: int):
        self.inputBatch = Matrix(inputNeuronCount, trainingBatchSize)
        self.outputBatch = Matrix(outputNeuronCount, trainingBatchSize)

        self.trainingBatchSize = trainingBatchSize
        self.inputNeuronCount = inputNeuronCount
        self.outputNeuronCount = outputNeuronCount

        self.ifs = open("training.txt", "r")
        self.floats = self.ifs.read().replace("\n", " ").replace("  ", " ").split(" ")

    def __del__(self):
        self.ifs.close()

    def GetBatchFloat(self):
        for flt in self.floats:
            yield float(flt)

    def GetNextBatch(self):
        for c in range(self.trainingBatchSize):
            for r in range(self.inputNeuronCount):
                # input is a Python keyword
                self.inputBatch.data[r][c] = self.GetBatchFloat().__next__()

            for r in range(self.outputNeuronCount):
                # Same here
                self.outputBatch.data[r][c] = self.GetBatchFloat().__next__()

    def GetInputBatch(self):
        return self.inputBatch

    def GetOutputBatch(self):
        return self.outputBatch

    def PrintInputBatch(self):
        self.inputBatch.PrintMatrix()

    def PrintOutputBatch(self):
        self.outputBatch.PrintMatrix()
