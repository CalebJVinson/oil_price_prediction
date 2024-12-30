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
$$input_t = \sigma(w_i \cdot [h_(t-1), x_t] + b_i)$$

# Echo State Network (ESN) - Reservoir model

For the comparison model we use the ESN Reservoir model

# LSTM v. ESN Model

\begin{table}[h!]
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Feature}            & \textbf{Echo State Network (ESN)} & \textbf{LSTM}                    \\ \midrule
\textbf{Training Method}    & Solely Trains Output   & Each Weight in model is trained          \\
\textbf{Changes}   & Reservoir is composed of fixed nodes           & Utilizes gates to control changes in nodes and retention       \\
\textbf{Computational Cost} & Lower computation due to fixed nodes                               & Readjusts the nodes to better fit prediction                             \\
\textbf{Long-Term Memory}   & Limited                           & Excellent                        \\
\textbf{Data Requirements}  & Works with small datasets         & Requires larger datasets         \\
\textbf{Adaptability}       & Less adaptable                    & Highly adaptable                 \\
\textbf{Use Case}           & Moderate temporal dependencies    & Complex and long-term dependencies \\ \bottomrule
\end{tabular}
\caption{Comparison of ESN and LSTM for Time Series Tasks}
\label{tab:esn_vs_lstm}
\end{table}
