import torch.nn as nn

class ArnieClassification(nn.Module):
    
    """
    A custom three layer NN for traning on the Arnie dataset.
    """
    
    def __init__(self):
        super().__init__()

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 output classes for binary classification
        )

    def forward(self, x):
        # Pass input through layer 1
        x = self.layer1(x)
        # Pass input through layer 2
        x = self.layer2(x)
        # Pass input through layer 3
        x = self.layer3(x)
        # Flatten the output
        x = x.view(x.size(0), -1)
        # Pass through the fully connected layers
        x = self.fc(x)
        return x
