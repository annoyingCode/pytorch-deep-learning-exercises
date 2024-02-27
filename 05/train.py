# YOUR CODE HERE

"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import argparse
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Initializing argument parser and adding command-line arguments
parser = argparse.ArgumentParser(description="The programs takes in hyper-parameters from CLI and train and evaluate the TinyVGG model.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs to train the model for.")
parser.add_argument("--data", default="./data", type=str, help="Location where training and testing data is stored.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to train the model on for single iteration.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for the optimizer.")
parser.add_argument("--num_filters", default=10, type=int, help="Number of filters (kernels) in a single convolutional layer.")
args = parser.parse_args()
data_base_dir = args.data

if data_base_dir == "./data" or not os.path.isdir(data_base_dir):
    print("Please specify the target dataset.")
    exit(0)

# Setup hyperparameters
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.num_filters
LEARNING_RATE = args.lr

# Setup directories
train_dir = data_base_dir + "/train"
test_dir = data_base_dir + "/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
if __name__ == "__main__":
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
