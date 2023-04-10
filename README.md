# NeuralNetExplorers: Discover the Magic of Neural Networks

Discover the magic of neural networks with NeuralNetExplorers, an educational repository focused on simplifying the understanding of neurons and neural networks using an engaging species classification example.

## Dataset

The dataset contains information about magical creatures called Fluffies and Spikies. The model will use the creature's height and color as features to determine which species the creature belongs to.

The dataset consists of the following features:

- Height
- Color (encoded as numeric values)
  - Red: 1
  - Blue: 2
  - Green: 3
  - Yellow: 4

The species are represented as:

- Spikies: 1
- Fluffies: 2

## Usage

1. Clone the repository: `git clone https://github.com/yourusername/NeuralNetExplorers.git`
2. Install the required dependencies: `pip install numpy pandas`
4. Run the Neuron code to train the neuron: `python neuron.py`

## Neuron Code Overview

The neuron code consists of a `Neuron` class that implements the following methods:

- `__init__`: Initializes the neuron's weights and bias randomly.
- `sigmoid_activation`: Computes the sigmoid activation function for a given input.
- `sigmoid_derivative`: Computes the derivative of the sigmoid function for a given input.
- `binary_cross_entropy_loss`: Computes the binary cross-entropy loss for a given prediction and true label.
- `backpropagation`: Performs backpropagation to update the neuron's weights and bias.
- `train`: Trains the neuron on the given dataset for a specified number of epochs and learning rate.
- `forward_pass`: Performs a forward pass on the given input and computes the loss if the true label is provided.

The code reads the dataset, shuffles it, and trains the neuron model using the dataset. The model's performance is evaluated based on the mean loss for each epoch.

## Contributing

Feel free to open issues or submit pull requests if you find any bugs or want to improve the project. Your contributions are welcome!
