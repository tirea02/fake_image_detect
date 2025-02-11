import os

# Get the absolute path of the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define dataset paths globally
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets/real_vs_fake/real-vs-fake")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# Subdirectories for training, validation, and test sets
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
TEST_DIR = os.path.join(DATASET_PATH, "test")

print(f"Dataset Path Loaded: {DATASET_PATH}")
