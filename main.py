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


def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))


def mse_prime(y, y_pred):
    return 2 * (y_pred - y) / np.size(y)


data = pd.read_csv("./train.csv").values
np.random.shuffle(data)
data = data[:100]


st.title("My Neural Network")
st.subheader("Hello")
# pic = st.slider("Number of Picture for Training", 100, 200, 150, 10)
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
num_layers = st.slider("Number of Hidden Layers", 1, 9, 1) + 1

image_size = 28
# User input for number of nodes
# nodes = st.slider('Number of Nodes', 10, 16, 10)
nodes = 10

# User input for choice of activation function
activation = st.selectbox("Activation Function", ["Tanh", "ReLU", "Sigmoid"])
neurons = []
activations = []
neurons.append(image_size**2)
neurons.append(nodes)
activations.append(eval(f"{activation}()"))

for _ in range(num_layers - 1):
    neurons.append(nodes)
    activations.append(eval(f"{activation}()"))
network = Network(neurons, activations)

# Add trend button to start
if st.button("Start"):
    start_time = time.time()
    # write error to .csv file
    plot_data = np.array(list(network.gradient_descent(X, Y, epochs, learning_rate)))

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
        y_true = np.eye(nodes)[y].T.reshape(-1, 1)
        loss += mse(y_true, output)
    loss /= len(X)
    st.write("# Loss: ", round(loss, 5))

    for layer in network.layers[::2]:
        base_shape = layer.learning_weights[0].shape

        weights = []

        base = np.zeros(base_shape)
        for idx, weight in enumerate(layer.learning_weights):
            base += weight
            if idx % 100 == 0:
                weights.append(base / 100)
                base = weight

        weights = weights[:: int(epochs / 100)]
        weights = [weight * 10 for weight in weights]

        norm_weight = [weight / np.linalg.norm(weight) for weight in weights]
        fig, ax = plt.subplots()

        heat = ax.matshow(norm_weight[0])
        the_plot = st.pyplot(fig)

        for i in range(len(norm_weight)):
            heat.set_data(norm_weight[i])
            ax.set_title(f"epoch : {i}")
            the_plot.pyplot(fig)
            time.sleep(0.01)

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
