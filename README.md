For this project, I was working from my coursework series in mathematical computing. When considering the model, I wanted to use neural networks.

# Choice of Nueral Network

There are three broad neural network classes: *Feedforward*, *convolutional*, and *recurrent*.

We will use a recurrent network. However, we will justify this decision by defining the model and use cases of each model type.

## Feedforward Neural Network (FNN)

This type of neural network does not maintain memory and is best for tabular data as compared to time and image data. The FNNs are a linear net where there is a single layer of output nodes with linear actiavetion. These result in outputs resulting from a series of weights which are multiplied from the prior nodes before minimizing the errors.

## Convolutional Neural Network (CNN)

This model is a modification of the FNN that is ideal for imagery, via pooling. Since we are working with time-series, this does not make sense.
## Recurrent Neural Network (RNN)

The recurrent model differs from FNNs in that the data flows in multiple directions allowing for prior state connections, making it better for handling temporal and sequential dependencies. Additionally, holding these states retains memories in "hidden states". Lastly, this model is more complex than the FNN and is subject to issues through the vanishing and exploding gradient problem.

***This is why we utilize the Recurrent Neural Network model.***

So, when considering an RNN, there are particular approaches that can be taken to determine the memory mechanisms.

# Long Short-Term Memory Model (LSTM)

For this project I used the LSTM model to predict prices in the West Texas Intermediate (WTI) Crude Oil Futures [CL=F]. To model this, I used a straightforward split of 70% training data and 30% testing data over the available periods. For the LSTM model the goal is to track the memory along with three additional items: inputs, outputs, and forget gates. 

The memory mechanisms are what primarily differentiate this model as the forget gates control the memory retention. These gates follow the logit function, or standardized logistic, by assigning sigmoidal activation functions that round upward to 1 for information identified as helpful to the prediction over time. This is helpful as we train down the set since this accumulation of short term memories represents the "Long" in LSTM.
