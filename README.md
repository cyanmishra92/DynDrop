# Dynamic Dropout Techniques in LeNet5 with MNIST

This project demonstrates the integration of various dynamic dropout techniques into the LeNet5 model for training on the MNIST dataset. The system is designed to be flexible, allowing easy experimentation with different dropout methods through a configuration-driven approach.

## Project Structure

- `lenet_model.py`: Contains the definition of the LeNet5 model with hooks for dynamic dropout.
- `main.py`: The main script to run the training and testing of the model on MNIST.
- `dropout_factory.py`: A factory module that provides a method to instantiate dropout classes based on a string identifier.
- `config.ini`: Configuration file to set hyperparameters, the dropout method, and its parameters for the entire network.
- Various dropout method files:
  - `gradient_dynamic_dropout.py`
  - `l2_dynamic_dropout.py`
  - `optimal_brain_damage_dropout.py`
  - `fmre_dynamic_dropout.py`
  - `learning_sparse_masks_dropout.py`
  - `neuron_shapley_dropout.py`
  - `taylor_expansion_dropout.py`
- `README.md`: This file, providing documentation for the project.

## Dynamic Dropout Techniques

Each dropout file implements a specific technique for dynamically adjusting dropout during training:

1. **Gradient Dynamic Dropout (`gradient_dynamic_dropout.py`)**: Adjusts dropout levels based on gradient magnitudes.
2. **L2 Dynamic Dropout (`l2_dynamic_dropout.py`)**: Uses the L2 norm of weights to influence dropout rates.
3. **Optimal Brain Damage Dropout (`optimal_brain_damage_dropout.py`)**: Employs a simplified version of the Optimal Brain Damage pruning method to adjust dropout.
4. **Feature Map Reconstruction Error Dropout (`fmre_dynamic_dropout.py`)**: Uses the reconstruction error of feature maps to adjust dropout.
5. **Learning Sparse Masks Dropout (`learning_sparse_masks_dropout.py`)**: Adapts dropout masks as learnable parameters within the network.
6. **Neuron Shapley Value Dropout (`neuron_shapley_dropout.py`)**: Applies the concept of Shapley values from game theory to assess neuron importance for dropout.
7. **Taylor Expansion Dropout (`taylor_expansion_dropout.py`)**: Uses Taylor expansion to evaluate the impact of neurons on loss for dropout adjustments.

## Configuration

The `config.ini` file allows you to set the model's hyperparameters and select which dropout method to apply across the entire network. Modify the `[DEFAULT]` section to change training settings like batch size or epochs. The `[Dropout]` section lets you choose the dropout method and its specific parameters.

Example `config.ini`:

```ini
[DEFAULT]
Epochs = 10
LearningRate = 0.001
BatchSize = 64

[Dropout]
Method = gradient_dynamic_dropout
InitialRate = 0.1
MaxRate = 0.5

