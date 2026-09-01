"""
File contains models used for training MRSI QC labeling.

CNN_QC - CNN Model containing 6 convolution layers 
                Followed by two fully connected layers and final lable. 

ICNN_QC - Inception CNN model. 
        The ICNN consisted of 2 convolutional layers with max pooling
        Two inception module layers followed by max pooling, 
        two fully connected layers, and a final output layer.

InceptionBlock - The inception block definition.

"""

import torch
import torch.nn as nn

class CNN_QC(nn.Module):
    """
    1D CNN for binary classification of MRSI QC labels.

    Input:
        x: (batch_size, 1, 850)
        in forward() method, can use the return_features boolean to 
        return learned features to be forwarded to RF classifier.       

    Output:
        logits: (batch_size, 2)

    Model Architecture:
        Six convolution layers, folowed by flattening and two fully connectedc
        dense layers: 
        
            Input
              |
            Conv1D(64, kernel=5, stride=1)
            ReLU
            MaxPool1D(kernel=2, stride=2)
              |
            Conv1D(64, kernel=5)
            ReLU
            MaxPool1D(kernel=2, stride=2)
              |
            Conv1D(128, kernel=5)
            ReLU
            MaxPool1D(kernel=2, stride=2)
              |
            Conv1D(128, kernel=5)
            ReLU
            MaxPool1D(kernel=2, stride=2)
              |
            Conv1D(64, kernel=5)
            ReLU
            MaxPool1D(kernel=2, stride=2)
              |
            Conv1D(64, kernel=5)
            ReLU
            MaxPool1D(kernel=2, stride=2)
              |
            Flatten
              |
            Fully Connected(64)
            ReLU
            Dropout(0.25)
              |
            Fully Connected(64)
            ReLU
            Dropout(0.25)
              |
            Fully Connected(2)

       Final output layer with 2 units for binary classification

    Feature extraction: 
        If return_features=True, the model returns the 64-dimensional
        representation from the second fully connected layer, immediately
        before the final classification layer.

        These features can then be forwarded to RF classifier 
 
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding='same'
        )

        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding='same')
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding='same')
        self.conv5 = nn.Conv1d(128, 64, kernel_size=5, padding='same')
        self.conv6 = nn.Conv1d(64, 64, kernel_size=5, padding='same')

        # Other layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Dense layers
        self.dense1 = nn.Linear(64 * (850 // 2**6), 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x, return_features=False):

        # CNN
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.pool(self.relu(self.conv6(x)))

        # Flatten
        x = self.flatten(x)

        # Dense layers
        x = self.relu(self.dense1(x))
        x = self.dropout(x)

        features = self.relu(self.dense2(x))

        # Return the learned 64-dimensional representation
        # --> this is used to forward features to RF classifier
        if return_features:
            return features

        x = self.dropout(features)
        x = self.dense3(x)

        return x

class InceptionBlock(nn.Module):
    """
    Defines the Inception block

    Inception block is comprised of four parallel branches,
    which are then concatenated along channel dim. Branches:
            1x1 convolution
            1x1 projection -> 3x3 convolution
            1x1 projection -> 5x5 convolution
            3x3 max pooling -> 1x1 projection

            Each convolution is followed by:
                BatchNorm1d -> LeakyReLU

    Inputs:
        in_channels : int
            Number of input channels.

        filters : int
            Number of output channels produced by each branch.

    Output:
        torch.Tensor
            Concatenation of the 4 branches
            Shape: (batch_size, filters * 4, signal_length)
    """

    def __init__(self, in_channels, filters):
        super().__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel_size=1, padding='same'),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(inplace=True)
        )

        # Branch 2: 1x1 projection -> 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel_size=1, padding='same'),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(filters, filters, kernel_size=3, padding='same'),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(inplace=True)
        )

        # Branch 3: 1x1 projection -> 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, filters, kernel_size=1, padding='same'),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(filters, filters, kernel_size=5, padding='same'),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(inplace=True)
        )

        # Branch 4: MaxPool -> 1x1 projection
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(in_channels, filters, kernel_size=1, padding='same'),
            nn.BatchNorm1d(filters),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenate the four branches and return
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class ICNN_QC(nn.Module):
    """
    1D Inception CNN for binary classification of MRSI QC labels.

    Model Architecture:

        Input
          |
        Conv1D(64, kernel=7, stride=2)
        BatchNorm
        LeakyReLU
        MaxPool1D(kernel=3, stride=2)
          |
        Conv1D(128, kernel=3)
        BatchNorm
        LeakyReLU
        MaxPool1D(kernel=3, stride=2)
          |
        InceptionBlock(64)
          |
        InceptionBlock(120)
          |
        MaxPool1D(kernel=3, stride=2)
          |
        Flatten
          |
        Fully Connected(64)
        ReLU
        Dropout(0.25)
          |
        Fully Connected(64)
        ReLU
        Dropout(0.25)
          |
        Fully Connected(2)

    Input:
        x : torch.Tensor
            Shape: (batch_size, 1, signal_length)

            For the current MRSI QC data, expects 850 spectral points

    Output:
        torch.Tensor
            Shape: (batch_size, 2)

            The two values are class logits.
            Apply torch.softmax() during inference to get class probabilities.

    Feature extraction: 
        If return_features=True, the model returns the 64-dimensional
        representation from the second fully connected layer, immediately
        before the final classification layer.

        These features can then be forwarded to RF classifier 
        
    """

    def __init__(self, signal_length=850):
        super().__init__()

        # Stem before inception modules
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Inception modules

        # 4 branches x 64 channels = 256 channels
        self.inception1 = InceptionBlock(in_channels=128, filters=64)

        # 4 branches x 120 channels = 480 channels
        self.inception2 = InceptionBlock(in_channels=256, filters=120)

        # Final pooling
        self.final_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Feature size calculation dynamic based on spectral points input
        with torch.no_grad():

            dummy = torch.zeros(1, 1, signal_length)
            dummy = self.stem(dummy)
            dummy = self.inception1(dummy)
            dummy = self.inception2(dummy)
            dummy = self.final_pool(dummy)

            flat_dim = dummy.flatten(1).shape[1]

        # Dense layers
        self.dense1 = nn.Linear(flat_dim, 64)
        self.dense2 = nn.Linear(64, 64)
        self.dense3 = nn.Linear(64,2)

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, return_features=False):

        # Inception layers
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.final_pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # dense layers
        x = self.relu(self.dense1(x))
        x = self.dropout(x)

        # This is the learned 64-dimensional representation
        features = self.relu(self.dense2(x))

        # Learned features to be forwarded to RF classifier
        if return_features:
            return features

        # Final classification layer
        x = self.dropout(features)
        x = self.dense3(x)

        return x
