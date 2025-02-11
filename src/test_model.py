import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import DeepCNN  # Import the trained model structure

# ✅ Step 1: Set Correct Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Points to `root/`
MODEL_PATH = os.path.join(PROJECT_ROOT,  "models", "deep_cnn_model.pth")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets", "sample_images")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "datasets", "predictions.txt")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found: {MODEL_PATH}")

print(f"✅ Model file found at: {MODEL_PATH}")

# ✅ Step 2: Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set model to evaluation mode

print("✅ Model loaded successfully")

# ✅ Step 3: Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def preprocess_image(image_path):
    """Loads and preprocesses an image for model testing."""
    image = Image.open(image_path).convert("RGB")  # Open image
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)


# ✅ Step 4: Predict All Images in `sample_images/`
def batch_predict(image_folder, results_file):
    """Predicts all images in the given folder and saves results."""
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"🚨 Image directory not found: {image_folder}")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("🚨 No images found in the directory!")
        return

    print(f"🔍 Found {len(image_files)} images. Running predictions...")

    predictions = []

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = preprocess_image(image_path)

        # Perform inference
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Get class labels
        class_labels = ["AI-Generated", "Real"]
        prediction = class_labels[predicted.item()]

        result = f"{image_file} → {prediction}"
        print(result)
        predictions.append(result)

    # Save results to file
    with open(results_file, "w") as f:
        f.write("\n".join(predictions))

    print(f"📄 Predictions saved to: {results_file}")


# ✅ Step 5: Run Batch Prediction
batch_predict(IMAGE_DIR, RESULTS_FILE)
