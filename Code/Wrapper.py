#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
from Network.Network import HomographyModel
import torch
from torchsummary import summary
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((filename, img))
    return images

# Stitch Images and Crop Black Spaces
def stitch_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)

    # Calculate the bounding box for the combined image
    all_corners = np.concatenate((corners1_transformed, corners2), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Translation matrix to shift the image into positive coordinates
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Warp img1 and create a base for the stitched image
    stitched_img = cv2.warpPerspective(img1, H_translation @ H, (x_max - x_min, y_max - y_min))

    # Overlay img2 on the stitched image
    overlay_x = translation_dist[0]
    overlay_y = translation_dist[1]
    stitched_img[overlay_y:overlay_y + h2, overlay_x:overlay_x + w2] = np.where(
        img2 != 0, img2, stitched_img[overlay_y:overlay_y + h2, overlay_x:overlay_x + w2]
    )

    # Crop black spaces
    gray_stitched = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_stitched, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    stitched_img_cropped = stitched_img[y:y + h, x:x + w]

    return stitched_img_cropped


def compute_homography_dlt(C_A, C_B):
    """
    Compute homography H using Direct Linear Transform (DLT)
    given original corners C_A and warped corners C_B.
    """
    # Compute homography using OpenCV's findHomography (DLT internally)
    H, _ = cv2.findHomography(C_A, C_B, method=0)  # method=0 -> DLT

    return H



def main():



    Parser = argparse.ArgumentParser()
    Parser.add_argument('--FilePath', dest='FilePath', default='../P1Ph2TestSet/Phase2Pano/tower/', help='Path to load images from')
    Parser.add_argument('--ModelPath', dest='ModelPath', default='../Checkpoints/99model.ckpt', help='Path to load trained model from')
    Parser.add_argument('--Method', dest='Method', default='supervised', help='homography method')
    Args = Parser.parse_args()
    FilePath = Args.FilePath
    ModelPath = Args.ModelPath
    Method = Args.Method

    # ModelPath = "../Checkpoints_run3/99model.ckpt"

    # Using supervised learning model
    model = HomographyModel().to(device)
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])

    """
    Read a set of images for Panorama stitching
    """
    # os.makedirs(output_folder, exist_ok=True)
    images = load_images_from_folder(FilePath)
    if len(images) < 2:
        print("Need at least two images for stitching.")
        return

    # Initialize stitching with the first image
    panorama = images[0][1]

    for i in range(1, len(images)):


        img1 = panorama
        img2 = images[i][1]

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        

        """
        Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
        """

        

        all_H_values = []

        for i in range(1):

            # Step 1: Define original patch corners C_A
            h, w = img1.shape[:2]
            ph, pw = (128,128)
            
            x_start = np.random.randint(0, w - pw)
            y_start = np.random.randint(0, h - ph)
            # x_start = w - pw
            # y_start = h - ph

            C_A = np.array([
                [x_start, y_start],
                [x_start + pw, y_start],
                [x_start, y_start + ph],
                [x_start + pw, y_start + ph]
            ], dtype=np.float32)

            patch_A = gray1[y_start:y_start+ph, x_start:x_start+pw]
            patch_B = gray2[y_start:y_start+ph, x_start:x_start+pw]

            stacked_patches = np.dstack((patch_A, patch_B))
            stacks = []
            stacks.append(torch.tensor(stacked_patches, dtype=torch.float32).permute(2, 0, 1) / 255.0)
            patches = torch.stack(stacks)

            patches = patches.to(device)

            # Step 2: Define H4Pt (simulated example, normally this comes from the model)
            output_tensor = model(patches)
            # Move to CPU and convert to NumPy
            H4Pt = output_tensor.detach().cpu().numpy().reshape(4, 2)
            # print(H4Pt)

            # Step 3: Compute new warped corner points C_B
            C_B = C_A + H4Pt

            # Step 4: Compute homography using DLT
            H_temp = compute_homography_dlt(C_A, C_B)

            all_H_values.append(H_temp)

            print("Estimated Homography Matrix:\n", H_temp)

        # Take average of all H values
        H = np.mean(all_H_values,axis=0)

        # print(f"Final H : {H}")

        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """

        panorama = stitch_images(img1,img2,H)

        cv2.imwrite(f"mypano_{i}.png",panorama)
        print("intermediate panorama saved!")
    
    
    cv2.imwrite("mypano.png",panorama)
    print("panorama saved!")
    


if __name__ == "__main__":
    main()
