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

# Long Short-Term Memory (LSTM) Model

For this project I used the LSTM model to predict prices in the West Texas Intermediate (WTI) Crude Oil Futures [CL=F]. To model this, I used a straightforward split of 70% training data and 30% testing data over 120 active trading days. For the LSTM model the goal is to track the memory along with three additional items: inputs, outputs, and forget gates. 

The memory mechanisms are what primarily differentiate this model as the forget gates control the memory retention. These gates follow an activation function, in this casesigmoidal activation functions that round upward to 1 for information identified as helpful to the prediction over time. This is helpful as we train down the set since this accumulation of short term memories represents the "Long" in LSTM.

We set up the three items as functions:

$$input_t = \sigma(w_i \cdot [h_{t-1}, x_t] + b_i)$$

$$output_t = \sigma(w_o \cdot [h_{t-1}, x_t] + b_o)$$

$$forget_t = \sigma(w_f \cdot [h_{t-1}, x_t] + b_f)$$

where the $\sigma$ is the sigmoid activation function, the $h_(t-1)$ is the tracking info from the prior period, the $x_t$ is our input information, the $w_(...)$ is our weights, and the $b_(...)$ values are the biases experienced for each of the gates.

# Echo State Network (ESN) - Reservoir model

For the comparison model we use the ESN Reservoir model

