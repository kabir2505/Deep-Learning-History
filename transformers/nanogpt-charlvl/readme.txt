# Nanolevel GPT

This project implements a **Character-level GPT** model for text generation, trained on Paul Graham's essays. The model generates text based on character-level patterns and is trained to generate text by learning from the provided dataset.

## Directory Structure

- **data/**: Contains the dataset used for training (Paul Graham's essays).
- **src/**: Contains the source code for training and inference.
  - **train.py**: The script to train the model.
  - **model.py**: Defines the model architecture.
  - **data_loader.py**: Data loading utilities.
  - **engine.py**: Utilities for running the model and training processes.
- **gpt_(1).ipynb**: Jupyter notebook where the code was originally written for training and experimenting with the model.
- **model_weights_9000.pth**: The trained model weights.
- **requirements.txt**: Python dependencies for the project.

## Installation

Before running the project, ensure you have the required dependencies installed.

1. Clone the repository or download the project files.
2. Navigate to the project directory and install the required packages using `pip`:

   ```bash
   pip install -r requirements.txt