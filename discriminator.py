
import torch.nn as nn 

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1,64,4,2,1,bias =False),
            nn.LeakyReLU(negative_slope= 0.2,inplace=True),
            
            nn.Conv2d(64,128,4,2,1,bias =False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope= 0.2,inplace=True),

            nn.Conv2d(128,256,4,2,1,bias =False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope= 0.2,inplace=True),

            nn.Conv2d(256,512,4,2,1,bias =False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope= 0.2,inplace=True),

            nn.Conv2d(512,1,4,1,0,bias =False),
            nn.Sigmoid()
        )

    def forward(self,x):
        output = self.main(x)
        return output.view(-1) #creating into vector<flattern>
