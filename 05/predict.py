
# YOUR CODE HERE
import model_builder
import sys
import torch
import torchvision

from torchvision import transforms

if len(sys.argv) < 2:
    print("Please an image location to run inference on.")
    exit(0)

MODEL_PATH = "./models/05_going_modular_script_mode_tinyvgg_model.pth"
CLASSES = ['pizza', 'steak', 'sushi']

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=3
).to(device)

# Load the state of the model (trained parameters)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load in custom image and convert the tensor values to float32
custom_image = torchvision.io.read_image(str(sys.argv[1])).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255. 

# Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image).unsqueeze(dim=0).to(device)

with torch.inference_mode():
    custom_image_pred = model(custom_image_transformed)
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    print(f"Prediction probabilities: {custom_image_pred_probs}")

    # Convert prediction probabilities -> prediction labels
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1).cpu().item()
    pred_label_class = CLASSES[custom_image_pred_label]
    print(pred_label_class)
