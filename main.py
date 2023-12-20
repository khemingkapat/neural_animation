from neural_network import *
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import HeatMap
import time
import networkx as nx


def draw_neural_net(network: Network):
    """
    Draw a neural network cartoon using matplotlib and networkx.

    :param layer_sizes: list of layer sizes, including input and output dimensionality
    """
    node_spacing = 1
    max_nodes = 14
    layer_sizes = network.neurons
    layer_sizes = list(map(lambda x: min([max_nodes, x]), layer_sizes))
    st.set_option("deprecation.showPyplotGlobalUse", False)  # disable warning
    network_plot = st.pyplot()

    for epoch in range(0, epochs, int(epochs / 10)):
        # Create a new figure
        G = nx.DiGraph()
        for i in range(len(layer_sizes) - 1):  # loop through each layer
            layer = network.layers[::2][i]
            base_shape = layer.learning_weights[0].shape
            weights = []
            base = np.zeros(base_shape)
            for idx, weight in enumerate(layer.learning_weights):
                base += weight
                if idx % 99 == 0:
                    weights.append(base / 100)
                base = weight

            weights = weights[:: int(epochs / 10)]
            weights = [weight * 10 for weight in weights]
            norm_weight = [weight / np.linalg.norm(weight) for weight in weights]
            for j in range(layer_sizes[i]):  # loop through each node in the layer
                for k in range(
                    layer_sizes[i + 1]
                ):  # loop through each node in the next layer
                    G.add_edge((i, j), (i + 1, k), weight=weights[i][k][j])
        pos = {}  # position of each node
        for i, layer_size in enumerate(layer_sizes):  # loop through each layer
            layer_height = (layer_size - 1) / 2.0  # calculate the height of the layer
            for j in range(layer_size):  # loop through each node in the layer
                pos[(i, j)] = [
                    i,
                    layer_height - j * node_spacing,
                ]  # set position of each node based on layer and index of node in layer

        weights = nx.get_edge_attributes(G, "weight").values()
        node_output = network.outputs[epoch]
        nodes = [node_output[l][n][0] * 10 for l, n in G.nodes]

        nx.draw(
            G,
            pos,
            with_labels=False,
            arrows=False,
            node_size=200,
            node_color=nodes,
            cmap=plt.cm.Blues,
            edge_color=weights,
            edge_cmap=plt.cm.coolwarm,
        )  # draw the neural network with black node color
        plt.title(f"Neural Network Graph Epoch {epoch}")  # set the title of the plot
        # plt.show() # display the plot of the neural network graph
        network_plot.pyplot()

    st.write(
        "This is a graph of the neural network, the red edges are the edges from the input layer to the hidden layer, the blue edges are the edges from the hidden layer to the hidden layer, and the green edges are the edges from the hidden layer to the output layer. "
    )


def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))


def mse_prime(y, y_pred):
    return 2 * (y_pred - y) / np.size(y)


data = pd.read_csv("./train.csv").values
np.random.shuffle(data)
data = data[:100]


X = data[:, 1:].reshape(-1, 784, 1) / 255
Y = data[:, 0].reshape(-1, 1, 1)

st.title("My Neural Network")
st.image("https://img.mit.edu/files/images/202211/MIT-Neural-Networks-SL.gif")
st.write(
    """
## What is Neural Network ?

A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In essence, neural networks are used to approximate functions that can depend on a large number of inputs and are generally unknown.

Neural networks are a subset of machine learning and are at the heart of deep learning algorithms. They are called "neural" because they are designed to mimic neurons in the human brain. A neuron takes inputs, does some processing, and produces one output. Similarly, a neural network takes a set of inputs, processes them through hidden layers using weights that are adjusted during training, and outputs a prediction representing the combined input signal.

The neural network in this app is being used to recognize handwritten digits, a classic problem in machine learning. The network is trained on a dataset of handwritten digits and their corresponding labels, and it learns to map the input images to the correct digit.
"""
)
st.subheader(
    "This is a neural network that We made from scratch using Python and NumPy, Let's train a neural network to recognize handwritten digits!"
)


# User input for number of epochs and hidden layers
st.write(
    "Select the number of epochs (how many time we train the network), higher means more accurate but longer training time"
)
epochs = st.slider("Number Epochs ", 0, 1000, 50, 50)
# epochs = 10**epochs


# User input for learning rate
st.write(
    "learning rate, higher learning rate might help on training time but accuracy will be reduced"
)
learning_rate = st.slider("Learning Rate 10 power by", 0.1, 1.5, 0.4, 0.1)
learning_rate = 10**learning_rate


# User input for number of hidden layers
st.write("Number of hidden layer, might help on accuracy or might not ???")
num_layers = st.slider("Number of Hidden Layers", 1, 5, 1)
st.write("Number of neurons in each hidden layer(s)")
neurons = [
    st.slider(f"Number of neurons in Hidden Layer {i+1}", 2, 16, 8, 2)
    for i in range(num_layers)
]


image_size = 28

# User input for choice of activation function
st.write("Select the activation function, Tanh is the default")
st.write(
    """
## Activation Functions

1. **Tanh (Hyperbolic Tangent):**
   - Outputs values between -1 and 1.
   - Symmetric around the origin.
   - Smooth and differentiable.
   - Often used in hidden layers of neural networks.

2. **ReLU (Rectified Linear Unit):**
   - Outputs the input directly if it is positive, otherwise outputs 0.
   - Computationally efficient and helps alleviate the vanishing gradient problem.
   - Introduces sparsity in the network by zeroing out negative values.
   - Commonly used in hidden layers of deep neural networks.

3. **Sigmoid:**
   - Outputs values between 0 and 1.
   - Maps the input to a probability-like output.
   - Useful for binary classification problems.
   - Suffers from the vanishing gradient problem for very large or small inputs.
   - Often used in the output layer for binary classification tasks.
"""
)

activation = st.selectbox("Activation Function", ["Tanh", "ReLU", "Sigmoid"])
activations = []
activations.append(eval(f"{activation}()"))

neurons = [28**2] + neurons + [10]

for _ in range(num_layers):
    activations.append(eval(f"{activation}()"))
network = Network(neurons, activations)

# Add trend button to start
if st.button("Start"):
    start_time = time.time()
    plot_data = np.array(list(network.gradient_descent(X, Y, epochs, learning_rate)))
    st.write(f"### Total Training Time : {time.time() - start_time:.2f}")

    st.write(f"# Network predicting for number : `{Y[0,0,0]}`")
    draw_neural_net(network)

    # Visualize accuracy and loss using matplotlib
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_data[:, 0], plot_data[:, 1], "r", label="Error Rate")
    ax.set_title("Error Rate")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Error percentage")
    ax.legend()

    # Show loss graph
    st.pyplot(fig)

    # Calculate accuracy of model
    accuracy = 0
    for x, y in zip(X, Y):
        output = network.forward(x)
        accuracy += (np.argmax(output) == y).mean()
    accuracy /= len(X)
    st.write("# Avg. Accuracy: ", round(accuracy * 100, 5), "%")

    # Calculate loss of model
    loss = 0
    for x, y in zip(X, Y):
        output = network.forward(x)
        y_true = np.eye(10)[y].T.reshape(-1, 1)
        loss += mse(y_true, output)
    loss /= len(X)
    st.write("# Loss: ", round(loss * 100, 5), "%")

    for lidx, layer in enumerate(network.layers[::2]):
        st.write(f"# Layer {lidx+1}")
        st.write(
            f"This layer have `{layer.input_size}` inputs to `{layer.output_size}` outputs"
        )
        st.write(
            f"Total of `{layer.input_size * layer.output_size}` weights and `{layer.output_size}` biases"
        )
        base_shape = layer.learning_weights[0].shape

        weights = []

        base = np.zeros(base_shape)
        for idx, weight in enumerate(layer.learning_weights):
            base += weight
            if idx % 99 == 0:
                weights.append(base / 100)
                base = weight

        weights = weights[:: int(epochs / 10)]
        weights = [weight * 10 for weight in weights]

        norm_weight = [weight / np.linalg.norm(weight) for weight in weights]
        if lidx == 0:
            norm_weight = [
                weight.reshape(layer.output_size, image_size, image_size).tolist()
                for weight in norm_weight
            ]

            fig, axs = plt.subplots(2, int(len(norm_weight[0]) / 2), figsize=(20, 10))
            heat_plot = st.pyplot(fig)
            for i in range(len(norm_weight)):
                for j in range(layer.output_size):
                    ax = axs[j % 2, j // 2]
                    heat = ax.matshow(norm_weight[i][j], cmap=plt.cm.coolwarm)

                fig.suptitle(f"epoch : {i*(int(epochs/10))}", fontsize=16)
                heat_plot.pyplot(fig)

        else:
            fig, ax = plt.subplots()

            heat = ax.matshow(norm_weight[0], cmap=plt.cm.coolwarm)
            heat_plot = st.pyplot(fig)

            for i in range(len(norm_weight)):
                heat.set_data(norm_weight[i])
                ax.set_title(f"epoch : {i*int(epochs/10)}")
                heat_plot.pyplot(fig)
                time.sleep(0.1)

        # bias part
        base_shape = layer.learning_biases[0].shape

        biases = []

        base = np.zeros(base_shape)
        for idx, bias in enumerate(layer.learning_biases):
            base += bias
            if idx % 99 == 0:
                biases.append(base / 100)
                base = bias

        biases = biases[:: int(epochs / 10)]
        biases = [bias * 10 for bias in biases]

        norm_bias = [(bias / np.linalg.norm(bias)).reshape(1, -1) for bias in biases]
        x = [f"bias {i}" for i in range(1, len(norm_bias[0].tolist()[0]) + 1)]
        chart = st.empty()
        # my_cmap = plt.get_cmap("inferno")
        # rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

        for i in range(len(norm_bias)):
            data = pd.DataFrame(norm_bias[i].reshape(-1, 1), index=x)
            chart.bar_chart(data)
            time.sleep(1)

    #
    # Show time taken to train
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
