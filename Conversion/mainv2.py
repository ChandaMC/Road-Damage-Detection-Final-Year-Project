import torch
import json
import sys

# Load metadata
with open("tm-my-image-model/metadata.json", "r") as f:
    metadata = json.load(f)

# Ensure that "imageSize" exists in metadata
if "imageSize" not in metadata:
    print("Error: 'imageSize' key not found in metadata.json")
    sys.exit(1)

# Ensure that "imageSize" has a valid value
image_size = metadata["imageSize"]
if not isinstance(image_size, int) or image_size <= 0:
    print("Error: 'imageSize' should be a positive integer")
    sys.exit(1)


# Define a class to encapsulate the model behavior
class TeachableMachineModel(torch.nn.Module):
    def __init__(self, metadata):
        super(TeachableMachineModel, self).__init__()

        # Determine the number of classes from the metadata
        if 'num_classes' in metadata:
            self.num_classes = metadata['num_classes']
        elif 'labels' in metadata:
            self.num_classes = len(metadata['labels'])
        else:
            raise ValueError("Unable to determine the number of classes from metadata")

        # Use "imageSize" as input size
        self.input_size = image_size

        # Define your model architecture based on the metadata
        self.fc = torch.nn.Linear(self.input_size, self.num_classes)

        # Load weights from weights.bin
        self.load_state_dict(torch.load("tm-my-image-model/weights.bin"))

    def forward(self, x):
        # Define the forward pass of your model
        x = self.fc(x)
        return x


# Create an instance of the TeachableMachineModel class
model = TeachableMachineModel(metadata)

# Optionally, switch to evaluation mode
model.eval()

# Convert the model to TorchScript
input_tensor = torch.ones((1, model.input_size), dtype=torch.float32)
traced_model = torch.jit.trace(model, input_tensor)

# Save the TorchScript model
traced_model.save("converted_model.torchscript.ptl")

print("Conversion to TorchScript complete!")
