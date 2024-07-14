import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, generator, num_classes=1000):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
        self.init_layers(generator)

    def init_layers(self, generator):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) and m != self.classifier[-1]:
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu', generator=generator)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif m == self.classifier[-1]:
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='linear', generator=generator)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
