import torchvision.datasets as dsets 
import torchvision.transforms as transforms  
import torchvision.utils as vutils
import torch.nn.parallel
import torch.optim as optim  
from torch.autograd import Variable 
from __future__ import print_function 
import torch 

from generator import Generator
from discriminator import Discriminator

batchSize = 64 
imageSize = 64 


transform = transforms.Compose([transforms.Scale(imageSize),
transforms.ToTensor(),
transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = dsets.MNIST(root='./data',download=True,transform=transform)

def initialiseWeights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0,0.2)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0,.02)
        model.bias.data.fill_(0)


generatorNN = Generator()
generatorNN.apply(initialiseWeights)
generatorNN.cuda()


discriminatorNN = Discriminator()
discriminatorNN.apply(initialiseWeights)
discriminatorNN.cuda()