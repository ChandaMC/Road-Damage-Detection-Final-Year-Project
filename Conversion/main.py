import torch
from yolov5.models.experimental import attempt_load
from torch.utils.mobile_optimizer import optimize_for_mobile


# Define a model wrapper to remove *args and **kwargs
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def export_torchscript(model_path, save_path):
    # Step 1: Load the YOLOv5 model
    device = torch.device('cpu')  # Use 'cuda' if you have a GPU
    model = attempt_load(model_path, map_location=device)
    model.eval()  # Set the model to evaluation mode

    # Step 2: Wrap the model
    wrapped_model = ModelWrapper(model)

    # Step 3: Convert the model to TorchScript using scripting (preferred for complex models)
    scripted_module = torch.jit.script(wrapped_model)

    # Step 4: Optimize for mobile (optional, only if you need to deploy on mobile)
    optimized_script_module = optimize_for_mobile(scripted_module)

    # Step 5: Save the TorchScript model
    optimized_script_module._save_for_lite_interpreter(save_path)
    print(f"TorchScript model saved at {save_path}")


if __name__ == "__main__":
    model_path = 'best.pt'  # Path to your YOLOv5 model
    save_path = 'best.torchscript.ptl'  # Path to save the TorchScript model
    export_torchscript(model_path, save_path)
