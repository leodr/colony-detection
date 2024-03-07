import torch.nn as nn


class CountingFCN(nn.Module):
    def __init__(self):
        super(CountingFCN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Reduce to 1520x1520
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Reduce to 760x760
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # Reduce to 380x380
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128, 256, kernel_size=5, stride=2, padding=2
            ),  # Reduce to 190x190
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Reduce to 95x95
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # Reduce to 48x48
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling to make it 1x1
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1)  # Final reduction to get the count
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.reshape(x.size(0), -1)
