from neural_network import *
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

        nx.draw(
            G,
            pos,
            with_labels=False,
            arrows=False,
            node_size=200,
            node_color="black",
            edge_color=weights,
            edge_cmap=plt.cm.inferno,
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


st.title("My Neural Network")
st.subheader("Hello")
# pic = st.slider("Number of Picture for Training", 100, 500, 250, 10)
# data = data[:pic]


X = data[:, 1:].reshape(-1, 784, 1) / 255
Y = data[:, 0].reshape(-1, 1, 1)


# User input for number of epochs and hidden layers
epochs = st.slider("Number of Epochs 10 power by", 2, 4, 3)
epochs = 10**epochs
# User input for learning rate
learning_rate = st.slider("Learning Rate 10 power by", -6, 0, -2)
learning_rate = 10**learning_rate
# User input for size of image
# image_size = st.slider('Image Size', 1, 28, 28)

# User input for number of hidden layers
num_layers = st.slider("Number of Hidden Layers", 1, 5, 1)
neurons = [
    st.slider(f"Number of neurons in layer{i+1}", 1, 15, 8) for i in range(num_layers)
]


image_size = 28
# User input for number of nodes
# nodes = st.slider('Number of Nodes', 10, 16, 10)
# nodes = 10

# User input for choice of activation function
activation = st.selectbox("Activation Function", ["Tanh", "ReLU", "Sigmoid"])
activations = []
activations.append(eval(f"{activation}()"))

neurons = [28**2] + neurons + [10]

for _ in range(num_layers):
    activations.append(eval(f"{activation}()"))
network = Network(neurons, activations)

# Add trend button to start
if st.button("Start"):
    print("started")
    start_time = time.time()
    plot_data = np.array(list(network.gradient_descent(X, Y, epochs, learning_rate)))

    print(neurons)
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
    st.write("# Loss: ", round(loss, 5))

    for lidx, layer in enumerate(network.layers[::2]):
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

            fig, axs = plt.subplots(2, 5, figsize=(20, 10))
            the_plot = st.pyplot(fig)
            for i in range(len(norm_weight)):
                for j in range(layer.output_size):
                    ax = axs[j % 2, j // 2]
                    heat = ax.matshow(norm_weight[i][j])

                fig.suptitle(f"epoch : {i}", fontsize=16)
                the_plot.pyplot(fig)

        else:
            fig, ax = plt.subplots()

            heat = ax.matshow(norm_weight[0])
            the_plot = st.pyplot(fig)

            for i in range(len(norm_weight)):
                heat.set_data(norm_weight[i])
                ax.set_title(f"epoch : {i}")
                the_plot.pyplot(fig)
                time.sleep(0.1)

    # Show Weights as a heatmaps using matplotlib.pyplot from layer to node
    # loop through every layer
    # for i, layer in enumerate(network.layers):
    #     if hasattr(layer, "weight"):
    #         st.subheader(
    #             "Heatmap of Weights from layer {} to layer {}".format(
    #                 int(i / 2), int(i / 2 + 1)
    #             )
    #         )
    #         if i == 0:  # If first layer
    #             data = layer.weight
    #             data = data.reshape(
    #                 nodes, image_size, image_size
    #             )  # Split data into nodes * image_size * image_size matrices
    #             data = data.tolist()
    #             # create heatmap for 1st layer to 2nd layer
    #             for j in range(nodes):
    #                 heatmap_data = pd.DataFrame(data[j])
    #                 heatmap_data.columns = [
    #                     str(k) for k in range(28)
    #                 ]  # Set column names as string numbers
    #                 heatmap_data.index = [
    #                     str(k) for k in range(28)
    #                 ]  # Set index names as string numbers
    #                 heatmap_data_list = heatmap_data.values.tolist()
    #                 heatmap_data_list = [
    #                     [k, l, heatmap_data_list[k][l]]
    #                     for k in range(28)
    #                     for l in range(28)
    #                 ]  # Adjust the range to 28
    #
    #                 # Show Weights as a heatmaps using matplotlib.pyplot
    #                 fig, ax = plt.subplots(figsize=(10, 10))
    #                 # set title of each heatmap
    #                 ax.set_title(
    #                     f"Heatmap of Weights from layer {i} to node {j} in layer {i+1}"
    #                 )
    #                 ax = sns.heatmap(heatmap_data, cmap="coolwarm")
    #                 st.pyplot(fig)
    #
    #         else:  # If not first layer, show heat map size of nodes * nodes
    #             data = layer.weight
    #             data = data.reshape(nodes, nodes)
    #             data = data.tolist()
    #             # create heatmap for Nnd layer to (N+1)rd layer
    #             heatmap_data = pd.DataFrame(data)
    #             heatmap_data.columns = [str(k) for k in range(nodes)]
    #             heatmap_data.index = [str(k) for k in range(nodes)]
    #             heatmap_data_list = heatmap_data.values.tolist()
    #             heatmap_data_list = [
    #                 [k, l, heatmap_data_list[k][l]]
    #                 for k in range(nodes)
    #                 for l in range(nodes)
    #             ]
    #
    #         # Show Weights as a heatmaps using matplotlib.pyplot
    #         fig, ax = plt.subplots(figsize=(10, 10))
    #         # set title of each heatmap
    #         ax.set_title(
    #             f"Heatmap of Weights from layer {int(i/2)} to layer {int(i/2+1)}"
    #         )
    #         ax = sns.heatmap(heatmap_data, cmap="coolwarm")
    #         st.pyplot(fig)
    #
    #         # Show Biases as a barchart using matplotlib.pyplot
    #         st.subheader(
    #             "Biases from layer {} to layer {}".format(int(i / 2), int(i / 2 + 1))
    #         )
    #         data = network.layers[i].bias
    #         data = data.reshape(nodes, 1)  # Split 10 into 10 1x1 matrices
    #         data = data.tolist()
    #         data = [j[0] for j in data]  # Convert to 1D list
    #
    #         # Show Biases as a barchart using matplotlib.pyplot
    #         fig, ax = plt.subplots(figsize=(10, 10))
    #         # set title of each barchart
    #         ax.set_title(
    #             "Biases from layer {} to layer {}".format(int(i / 2), int(i / 2 + 1))
    #         )
    #         bars = ax.bar([str(j) for j in range(nodes)], data)
    #
    #         # Add labels to each bar
    #         for bar in bars:
    #             height = bar.get_height()
    #             ax.annotate(
    #                 f"{height}",
    #                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 3),
    #                 textcoords="offset points",
    #                 ha="center",
    #                 va="bottom",
    #             )
    #
    #         st.pyplot(fig)
    #
    #     # print(i)
    #
    # # print("-" * 30 + "after trained" + "-" * 30)
    # # for x, y in list(zip(X, Y))[:20]:
    # #     output = network.forward(x)
    # #
    # #     print(f"actual y = {y}")
    # #     print(f"prediction = {np.argmax(output)}")
    # #     print("-" * 50)
    #
    # Show time taken to train
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
