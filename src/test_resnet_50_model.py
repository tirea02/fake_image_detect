import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.models import resnet50

# ✅ Step 1: Set Correct Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Points to `root/`
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "resnet50_best.pth")  # ✅ Ensure latest ResNet-50 model
IMAGE_DIR = os.path.join(PROJECT_ROOT, "datasets", "sample_images")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "datasets", "predictions_resnet.txt")

# ✅ Step 2: Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load ResNet-50 model structure
model = resnet50(weights=None)  # Load ResNet-50 architecture
model.fc = torch.nn.Linear(2048, 2)  # Ensure the final layer matches training setup

# ✅ Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found: {MODEL_PATH}")

# ✅ Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print(f"✅ Model file found and loaded: {MODEL_PATH}")

# ✅ Step 3: Define Image Preprocessing (Must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure the same size as training
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_image(image_path):
    """Loads and preprocesses an image for model testing."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"🚨 Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# ✅ Step 4: Predict All Images in `sample_images/`
def batch_predict(image_folder, results_file):
    """Predicts all images in the given folder and saves results."""
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"🚨 Image directory not found: {image_folder}")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

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
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # Get confidence scores
            _, predicted = torch.max(output, 1)

        # Get class labels
        class_labels = ["AI-Generated", "Real"]
        prediction = class_labels[predicted.item()]
        confidence = probabilities[predicted.item()].item() * 100  # Convert to percentage

        result = f"{image_file} → {prediction} ({confidence:.2f}%)"
        print(result)
        predictions.append(result)

    # Save results to file
    with open(results_file, "w") as f:
        f.write("\n".join(predictions))

    print(f"📄 Predictions saved to: {results_file}")

# ✅ Step 5: Run Batch Prediction
batch_predict(IMAGE_DIR, RESULTS_FILE)
