from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.pipeline = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2), #112
          nn.BatchNorm2d(64),
          nn.LeakyReLU(),
          nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), #56
          nn.BatchNorm2d(128),
          nn.LeakyReLU(),
          # nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), #28
          # nn.ReLU(),
          # nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),#14
          # nn.ReLU(),
          # nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2),#7
          # nn.ReLU(),
          nn.AvgPool2d(4,4),
          nn.Flatten(),
          nn.Linear(128*14*14,200)

          # Add more layers...
          #self.fc1 = nn.Linear(..) # 200 is the number of classes in TinyImageNet
        )

    def forward(self, x):
        # Define forward pass
        return self.pipeline(x)