import argparse
import torch
from Network.Network import HomographyModel
from torchvision.transforms import ToTensor
import os
import cv2
import numpy as np
import time


def load_dataset(data_path):
    """
    Load validation dataset from the specified path.
    Returns a list of tuples: (image_tensor, ground_truth_coordinates).
    """
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png')  # Supported file formats

    print(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation path '{data_path}' does not exist.")

    for filename in sorted(os.listdir(data_path)):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(data_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                gt_coords = np.array([[0, 0], [w, 0], [w, h], [0, h]])  # Placeholder ground truth
                images.append((ToTensor()(img), gt_coords))
            else:
                print(f"Warning: Unable to read image {img_path}. Skipping.")
        else:
            print(f"Skipping unsupported file: {filename}")

    if not images:
        raise ValueError(f"No valid images found in the dataset path: {data_path}")
    return images


def save_results(results_path, epe_values, runtimes):
    """
    Save EPE and runtime results to a text file.
    """
    with open(os.path.join(results_path, "test_results.txt"), "w") as f:
        f.write("Test Results:\n")
        f.write(f"Average End-Point Error (EPE): {np.mean(epe_values):.4f}\n")
        f.write(f"Average Runtime (ms): {np.mean(runtimes):.4f}\n")
        f.write("\nPer-Image Details:\n")
        for i, (epe, runtime) in enumerate(zip(epe_values, runtimes)):
            f.write(f"Image {i + 1}: EPE = {epe:.4f}, Runtime = {runtime:.4f} ms\n")


def test_model(args):
    """
    Test the trained model on the validation dataset and calculate metrics.
    """
    val_path = args.ValPath
    model_path = args.ModelPath
    results_path = args.ResultsPath

    os.makedirs(results_path, exist_ok=True)

    # Load dataset
    val_data = load_dataset(val_path)

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HomographyModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    epe_values = []
    runtimes = []
    print("Starting model evaluation...")

    for idx, (img, gt_coords) in enumerate(val_data):
        img = img.unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU
        gt_coords = torch.tensor(gt_coords, dtype=torch.float32).to(device)

        # Measure runtime
        start_time = time.time()
        with torch.no_grad():
            pred_coords = model(img).squeeze(0).cpu().numpy()  # Predicted coordinates
        end_time = time.time()

        runtime = (end_time - start_time) * 1000  # Convert to milliseconds
        runtimes.append(runtime)

        # Calculate End-Point Error (EPE)
        epe = np.linalg.norm(pred_coords - gt_coords.cpu().numpy())
        epe_values.append(epe)

        print(f"Image {idx + 1}: EPE = {epe:.4f}, Runtime = {runtime:.4f} ms")

    # Report Average EPE and Runtime
    average_epe = np.mean(epe_values)
    average_runtime = np.mean(runtimes)
    print(f"\nAverage End-Point Error (EPE): {average_epe:.4f}")
    print(f"Average Runtime (ms): {average_runtime:.4f}")

    # Save results
    save_results(results_path, epe_values, runtimes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Homography Estimation Model")
    parser.add_argument("--ValPath", default="C:\\cv\\YourDirectoryID_p1\\prangnekar_p1\\Phase2\\Data\\Val",
                        help="Path to validation data (default: C:\\cv\\YourDirectoryID_p1\\prangnekar_p1\\Phase2\\Data\\Val)")
    parser.add_argument("--ModelPath",
                        default="C:\\cv\\YourDirectoryID_p1\\prangnekar_p1\\Phase2\\Checkpoints\\best_model.ckpt",
                        help="Path to the trained model checkpoint (default: C:\\cv\\YourDirectoryID_p1\\prangnekar_p1\\Phase2\\Checkpoints\\best_model.ckpt)")
    parser.add_argument("--ResultsPath", default="C:\\cv\\YourDirectoryID_p1\\prangnekar_p1\\Phase2\\Results",
                        help="Path to save test results (default: C:\\cv\\YourDirectoryID_p1\\prangnekar_p1\\Phase2\\Results)")

    args = parser.parse_args()
    test_model(args)
