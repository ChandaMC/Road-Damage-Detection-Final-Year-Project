import torch

# Load the TorchScript model
model = torch.jit.load('yolov5s.torchscript.ptl')


# Function to print the output shape of a module
def print_output_shape(module, input):
    output = module(input)
    print(f"Output shape: {output.shape}")


# Traverse through the model and print the output shape of each module
def traverse_model(module, input):
    for name, submodule in module.named_children():
        print(f"Module: {name}")
        if isinstance(submodule, torch.nn.ModuleList):
            for idx, subsubmodule in enumerate(submodule):
                print(f"  Submodule {idx}:")
                traverse_model(subsubmodule, input)
        else:
            print_output_shape(submodule, input)


# Dummy input
dummy_input = torch.randn(1, 3, 416, 416)

# Traverse the model and print output shapes
traverse_model(model, dummy_input)
