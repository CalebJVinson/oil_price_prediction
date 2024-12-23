For this project, I was working from my coursework series in mathematical computing. When considering the model, I wanted to use neural networks.

# Choice of Nueral Network

There are three broad neural network classes: *Feedforward*, *convolutional*, and *recurrent*.

We will use a recurrent network. However, we will justify this decision by defining the model and use cases of each model type.

## Feedforward Neural Network (FNN)

This type of neural network does not maintain memory and is best for tabular data as compared to time and image data. The FNNs are a linear net where there is a single layer of output nodes with linear actiavetion. These result in outputs resulting from a series of weights which are multiplied from the prior nodes before minimizing the errors.

## Convolutional Neural Network (CNN)

This model is a modification of the FNN that is ideal for imagery, via pooling. Since we are working with time-series, this does not make sense.
## Recurrent Neural Network (RNN)

