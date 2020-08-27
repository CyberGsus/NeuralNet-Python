from NeuralNet import NeuralNet

if __name__ == "__main__":
    net: NeuralNet = NeuralNet(2, 3, 1)
    net.AddLayer(10)
    net.AddLayer(5)
    net.AddLayer(1)

    net.FeedForward()

    net.PrintInputLayer()
    net.PrintOutputLayer()

    for i in range(3000):
        net.FeedBack()
        net.FeedForward()

        net.PrintInputLayer()
        net.PrintOutputLayer()
