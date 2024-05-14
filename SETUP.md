### README.md for Setup and Execution


# Setup and Execution Guide for LeNet5 with Dynamic Dropout on MNIST

This guide provides detailed instructions on how to set up your environment and run the LeNet5 model with various dynamic dropout methods on the MNIST dataset.

## Prerequisites

Before you begin, ensure you have the following installed:
- Anaconda or Miniconda (Download from [here](https://www.anaconda.com/products/individual))

## Setting Up Your Environment

1. **Create a Conda Environment**:
   Open your terminal and create a new conda environment by running:

   ```bash
   conda create -n dyn_dropout python=3.8
   ```

   Activate the environment:

   ```bash
   conda activate dyn_dropout
   ```

2. **Install PyTorch and torchvision**:
   Install PyTorch and torchvision within the environment. Ensure to choose the command that fits your system (CPU or GPU). You can find the appropriate installation command for your system on the [PyTorch official site](https://pytorch.org/).

   For a typical CPU installation:

   ```bash
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

   For a typical GPU installation on Windows:

   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```

   Adjust the `cudatoolkit` version according to your CUDA version if applicable.

## Project Files Overview

Ensure you have the following files in your project directory:
- `lenet_model.py`: The model definition including dynamic dropout integration.
- `main.py`: Script to train and test the model.
- `dropout_factory.py`: Factory for instantiating dropout classes.
- Various dropout method scripts (`gradient_dynamic_dropout.py`, `l2_dynamic_dropout.py`, etc.).
- `config.ini`: Configuration file to adjust model parameters and dropout method.
- `README.md`: This guide.

## Configuring the Model

Edit the `config.ini` file to set the desired number of training epochs, learning rate, batch size, and the dropout method. For example:

```ini
[DEFAULT]
Epochs = 10
LearningRate = 0.001
BatchSize = 64

[Dropout]
Method = gradient_dynamic_dropout
InitialRate = 0.1
MaxRate = 0.5
```

## Running the Model

Once the environment is set up and the code is configured, run the model training and evaluation by executing:

```bash
python main.py
```

This command will start the training process using the settings specified in `config.ini`, and it will output training and testing performance metrics to the console.

## Saving and Loading Model

The trained model weights will be saved to `mnist_lenet.pth` at the end of training. You can modify `main.py` if you want to change the file name or path.

To load the model for further testing or deployment, use:

```python
model.load_state_dict(torch.load('mnist_lenet.pth'))
model.eval()  # Set the model to evaluation mode
```

## Extending the Project

To add new dropout methods:
1. Implement the method in a new Python script.
2. Update `dropout_factory.py` to include the new method.
3. Adjust `config.ini` to use the new method by changing the `[Dropout]` section.

Thank you for using this guide to set up and run the dynamic dropout integrated LeNet model on MNIST!
