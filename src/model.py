import torch
import torch.nn as nn

class DQN_CNN(nn.Module):
    def __init__(self, frame_skip = 4):
        super(DQN_CNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

         #w 82 - 7 + 2 * 3 / 2 + 1 = 41

         #w maxpool
         #w 41 / 2 = 20 out

         #h 136 - 7 + 2 * 3 / 2 + 1 = 68

        #h maxpool
        #h 68 / 2 = 34 out
        self.conv1 = nn.Conv2d(
            in_channels=frame_skip,
            out_channels=16,
            kernel_size=7,
            stride=2,
            padding=3,
        )


        #w maxpool
        #w 20 / 2 = 10 out

        #h maxpool
        #h 34 / 2 = 17 out
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )


        self.fc1 = nn.Linear(32 * 10 * 17, 128)
        self.fc2 = nn.Linear(128, 2)  # Assuming 2 actions: flap or not flap

        conv_layers = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
        )

        linear_layers = nn.Sequential(
            self.fc1,
            nn.LayerNorm(128),
            nn.ReLU(),
            self.fc2    
        )

        self.layers = nn.Sequential(
            conv_layers,
            nn.Flatten(),
            linear_layers
        )

    def forward(self, x):
        return self.layers(x)



