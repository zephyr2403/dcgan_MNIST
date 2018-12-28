import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100,out_channels=512,kernel_size=4,stride= 1,padding= 0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
    
    def forward(self,x):
        output = self.main(x)
        return output

c = Generator()