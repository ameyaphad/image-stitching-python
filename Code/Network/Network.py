"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import lightning as pl
import kornia # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def LossFn(predicted,groundtruth):
    ###############################################
    # Fill your loss function of choice here!
    criterion = torch.nn.MSELoss()
    loss = criterion(predicted,groundtruth)

    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    # loss = ...
    return loss


class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = HomographyNet()

    def forward(self, a):
        return self.model(a)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, patch_stack,H_stack):
        # patch_stack, H_stack = batch
        predicted_stack = self.model(patch_stack)
        loss = LossFn(predicted_stack,H_stack)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}



class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        input_size = 128
        final_feature_size = (input_size // 8) ** 2 * 128
        self.fc1 = nn.Linear(final_feature_size, 1024)
        self.fc2 = nn.Linear(1024, 8)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 8 values representing H4Pt
        
        return x



class TensorDLT(nn.Module):
    def __init__(self):
        super(TensorDLT, self).__init__()

    def forward(self, C_A, H4Pt):
        """
        Computes homography matrix from C_A and H4Pt using Kornia.
        C_A: (Batch, 4, 2) - original corner coordinates.
        H4Pt: (Batch, 4, 2) - predicted corner displacements.
        Returns:
        H: (Batch, 3, 3) - estimated homography matrices.
        """
        
        print(C_A)
        print(H4Pt)
        C_B = C_A + H4Pt  # Compute new corner positions

        B = C_A.shape[0]
        C_A = C_A.view(B, 4, 2)  # Reshape from [B, 8] to [B, 4, 2]
        C_B = C_B.view(B, 4, 2)

        # Compute homography using Kornia's DLT algorithm
        H = kornia.geometry.homography.find_homography_dlt(C_A, C_B)
        
        return H


def LossFn_unsup(patch_a_warped, patch_b):
    

    loss = F.l1_loss(patch_a_warped, patch_b)
    return loss



class Net(nn.Module):
    def __init__(self, InputSize=2):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()

        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        input_size = 128
        final_feature_size = (input_size // 8) ** 2 * 128
        self.fc1 = nn.Linear(final_feature_size, 1024)
        self.fc2 = nn.Linear(1024, 8)



        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(InputSize, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    
    def stn(self, x, H):
        """
        Spatial transformer network forward function.
        Takes in an image and a homography matrix.
        """
        # Process image through the localization network
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)

        # Use the regressor to obtain the affine matrix
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Optionally modify theta with the homography matrix
        # Homography matrix H is a 3x3 matrix, we need to convert it to affine (2x3)
        H_affine = H[:2, :]  # Take the top 2 rows for 2x3 affine

        # Combine the regressor theta with the homography matrix (H)
        theta = theta + H_affine  # Simple addition, you could use more complex operations if needed

        # Create grid from the final theta
        grid = F.affine_grid(theta, x.size())

        # Sample the image using the grid
        x = F.grid_sample(x, grid)

        return x


    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)  # 8 values representing H4Pt
        #############################
        return out

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, patch_stack,H_stack):
        # patch_stack, H_stack = batch
        predicted_stack = self.model(patch_stack)
        loss = LossFn(predicted_stack,H_stack)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}