#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code

"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel, LossFn, Net, LossFn_unsup, TensorDLT
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import csv
import gc
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


def apply_motion_blur(image, kernel_size=15, angle=0):
    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = cv2.warpAffine(kernel, 
                            cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1),
                            (kernel_size, kernel_size))
    kernel = kernel / kernel_size  # Normalize

    # Apply the motion blur kernel
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def add_occlusion(image, x=None, y=None, width=20, height=10):
    h, w = image.shape  # Only two dimensions for grayscale images

    # Choose random location if not provided
    if x is None:
        x = np.random.randint(0, w - width)
    if y is None:
        y = np.random.randint(0, h - height)

    intensity = 60

    # Apply occlusion (black patch by default, or specify intensity)
    occluded_image = image.copy()
    occluded_image[y:y+height, x:x+width] = intensity  # Set intensity value (0 for black)

    return occluded_image



def generate_homography_patch(image):

    h, w = image.shape[:2]
    ph, pw = (128,128)
    perturbation = 32
    translation = 4
    
    x_start = np.random.randint(0, w - pw)
    y_start = np.random.randint(0, h - ph)
    # x_start = w - pw
    # y_start = h - ph

    src_pts = np.array([
        [x_start, y_start],
        [x_start + pw, y_start],
        [x_start, y_start + ph],
        [x_start + pw, y_start + ph]
    ], dtype=np.float32)

    dst_pts = src_pts.copy()

    for i in range(4):

        shift = np.random.randint(-perturbation, perturbation)

        if shift == 0:
            shift = np.random.randint(-perturbation, perturbation)

        dst_pts[i][0] = dst_pts[i][0] + np.random.randint(-perturbation, perturbation)
        dst_pts[i][1] = dst_pts[i][1] + np.random.randint(-perturbation, perturbation)

    tx = np.random.randint(0,translation)
    ty = np.random.randint(0,translation)
    dst_pts += np.array([tx, ty], dtype=np.float32)  # Shift all points

    H_AB = cv2.getPerspectiveTransform(src_pts, dst_pts)

    if np.linalg.det(H_AB) == 0:
        return None, None, None

    H_BA = np.linalg.inv(H_AB)

    warped_image = cv2.warpPerspective(image, H_BA, (w, h))

    patch_A = image[y_start:y_start+ph, x_start:x_start+pw]
    patch_B = warped_image[y_start:y_start+ph, x_start:x_start+pw]

    black_pixels = (patch_B == 0).astype(np.uint8)  # Binary mask for black pixels
    black_area = np.sum(black_pixels)
    total_area = patch_B.shape[0] * patch_B.shape[1]

    if (black_area / total_area) > 0.1 :
        return None, None, None

    blurred_A = apply_motion_blur(patch_A, kernel_size=15, angle=15)
    blurred_B = apply_motion_blur(patch_B, kernel_size=15, angle=15)

    patch_A = add_occlusion(blurred_A)
    patch_B = add_occlusion(blurred_B)

    H4Pt = dst_pts - src_pts

    C_A = src_pts

    if len(image.shape) == 3:
        stacked_patches = np.dstack((patch_A, patch_B))
    else:
        stacked_patches = np.stack((patch_A, patch_B), axis=-1)

    # print(C_A)

    return stacked_patches, H4Pt, C_A


def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    """
    Inputs:
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainCoordinates - Coordinatess corresponding to Train
    NOTE that TrainCoordinates can be replaced by Val/TestCoordinatess for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    CoordinatesBatch - Batch of coordinates
    """
    PatchesBatch = []
    HomographyBatch = []
    CornersBatch = []
    ImagesBatch = []

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)

        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + ".jpg"
        ImageNum += 1

        
        ##########################################################
        I1 = np.float32(cv2.imread(RandImageName))
        I1 = cv2.resize(I1, (320, 240))
        ImagesBatch.append(torch.tensor(I1.flatten(),dtype=torch.float32))
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

        for i in range(1):

            stacked_patches, H4Pt, C_A = generate_homography_patch(I1)

            if stacked_patches is None:
                continue

            
            PatchesBatch.append(torch.tensor(stacked_patches, dtype=torch.float32).permute(2, 0, 1) / 255.0)
            HomographyBatch.append(torch.tensor(H4Pt.flatten(), dtype=torch.float32))
            CornersBatch.append(torch.tensor(C_A.flatten(),dtype=torch.float32))

    return (
        torch.stack(PatchesBatch).to(device, non_blocking=True),
        torch.stack(HomographyBatch).to(device, non_blocking=True),
        torch.stack(CornersBatch).to(device, non_blocking=True),
        torch.stack(ImagesBatch).to(device, non_blocking=True)
    )


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel().to(device)

    Optimizer = AdamW(model.parameters(), lr=1e-4)

    # Initialize CSV file for logging
    csv_file = "training_logs.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'Training Loss','Validation Loss'])


    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        loss_epoch = []
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch, CornersBatch, ImagesBatch = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
            )

            del CornersBatch, ImagesBatch
            # I1Batch, CoordinatesBatch = I1Batch.to(device), CoordinatesBatch.to(device)


            # Predict output with forward pass
            PredicatedCoordinatesBatch = model(I1Batch)
            LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)

            Optimizer.zero_grad(set_to_none=True)
            LossThisBatch.backward()
            Optimizer.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            result = model.validation_step(I1Batch,CoordinatesBatch)
            loss = result["val_loss"].item()
            # loss_epoch.append(result)
            loss_epoch.append(result["val_loss"].item())
            print(f"epoch:[{Epochs}] Iteration:[{PerEpochCounter}] loss:[{loss}]")

            # Explicitly delete tensors to free memory
            del I1Batch, CoordinatesBatch, PredicatedCoordinatesBatch, LossThisBatch
            gc.collect()
            torch.cuda.empty_cache()  # Optional: Clears GPU cache to free up memory
            
            
        loss_per_epoch = np.mean(loss_epoch)

        DirNamesPath = "./TxtFiles/DirNamesVal.txt"
        DirNamesVal = SetupDirNames(DirNamesPath)
        I1BatchVal, CoordinatesBatchVal = GenerateBatch(
                BasePath, DirNamesVal, TrainCoordinates, ImageSize, 1000
            )

        validation_loss = 0.0

        with torch.no_grad():
            I1BatchVal, CoordinatesBatchVal = I1BatchVal.to(device), CoordinatesBatchVal.to(device)
            validation_result = model.validation_step(I1BatchVal,CoordinatesBatchVal)
            validation_loss = validation_result["val_loss"].item()

        with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([Epochs, loss_per_epoch, validation_loss])

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": validation_loss,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")




def train_model(DirNamesTrain,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType):

    # Predict output with forward pass
    model = Net().to(device)
    model_tdlt = TensorDLT().to(device)
    

    Optimizer = AdamW(model.parameters(), lr=1e-4)

    

    # Initialize CSV file for logging
    csv_file = "training_logs_unsupervised.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'Training Loss','Validation Loss'])

    

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        loss_epoch = []
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            I1Batch, CoordinatesBatch,CornersBatch, ImagesBatch = GenerateBatch(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize
            )

            del CoordinatesBatch
            # I1Batch, CoordinatesBatch = I1Batch.to(device), CoordinatesBatch.to(device)

            Optimizer.zero_grad()
            
            # Predict H4Pt using HomographyNet
            H4Pt = model(I1Batch)

            # Compute Homography using TensorDLT
            # H = model_tdlt(CornersBatch, H4Pt.view(-1, 4, 2))
            H = model_tdlt(CornersBatch, H4Pt)

            # Warp P_A using computed H
            warped_P_A = model.stn(ImagesBatch,H)

            # Compute Photometric Loss
            loss = LossFn_unsup(warped_P_A, P_B)

            # Backpropagation
            loss.backward()
            optimizer.step()


            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            result = model.validation_step(I1Batch,CoordinatesBatch)
            loss = result["val_loss"].item()
            # loss_epoch.append(result)
            loss_epoch.append(result["val_loss"].item())
            print(f"epoch:[{Epochs}] Iteration:[{PerEpochCounter}] loss:[{loss}]")

            # Explicitly delete tensors to free memory
            del I1Batch, CoordinatesBatch, PredicatedCoordinatesBatch, LossThisBatch
            gc.collect()
            torch.cuda.empty_cache()  # Optional: Clears GPU cache to free up memory
            
            

        loss_per_epoch = np.mean(loss_epoch)

        DirNamesPath = "./TxtFiles/DirNamesVal.txt"
        DirNamesVal = SetupDirNames(DirNamesPath)
        I1BatchVal, CoordinatesBatchVal = GenerateBatch(
                BasePath, DirNamesVal, TrainCoordinates, ImageSize, 1000
            )

        validation_loss = 0.0

        with torch.no_grad():
            I1BatchVal, CoordinatesBatchVal = I1BatchVal.to(device), CoordinatesBatchVal.to(device)
            validation_result = model.validation_step(I1BatchVal,CoordinatesBatchVal)
            validation_loss = validation_result["val_loss"].item()

        with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([Epochs, loss_per_epoch, validation_loss])

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                "loss": validation_loss,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")




def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints_unsup/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=100,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )
    Parser.add_argument(
        "--Method",
        default="Supervised",
        help="Model to use while training, Default=Supervised",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType
    Method = Args.Method

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    if Method == "Supervised":

        TrainOperation(
            DirNamesTrain,
            TrainCoordinates,
            NumTrainSamples,
            ImageSize,
            NumEpochs,
            MiniBatchSize,
            SaveCheckPoint,
            CheckPointPath,
            DivTrain,
            LatestFile,
            BasePath,
            LogsPath,
            ModelType,
        )

    else:

        train_model(
            DirNamesTrain,
            TrainCoordinates,
            NumTrainSamples,
            ImageSize,
            NumEpochs,
            MiniBatchSize,
            SaveCheckPoint,
            CheckPointPath,
            DivTrain,
            LatestFile,
            BasePath,
            LogsPath,
            ModelType,
        )


if __name__ == "__main__":
    main()
