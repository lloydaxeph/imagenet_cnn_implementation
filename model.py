import torch
import torch.nn as nn


class CustomCNNModel(nn.Module):
    def __init__(self, n_classes: int = 1000):
        super(CustomCNNModel, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.model_features = nn.Sequential(
            self.conv1, nn.ReLU(True), nn.MaxPool2d((3, 3), (2, 2)),
            self.conv2, nn.ReLU(True), nn.MaxPool2d((3, 3), (2, 2)),
            self.conv3, nn.ReLU(True),
            self.conv4, nn.ReLU(True),
            self.conv5, nn.ReLU(True), nn.MaxPool2d((3, 3), (2, 2)),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        # The fully-connected layers have 4096 neurons each as stated in the paper.
        self.n_neurons = 4096
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, self.n_neurons),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.n_neurons, self.n_neurons),
            nn.ReLU(True),
            nn.Linear(self.n_neurons, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model_features(x)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)
