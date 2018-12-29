from __future__ import print_function 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms  
import torchvision.utils as vutils
import torch.nn.parallel
import torch.optim as optim  
from torch.autograd import Variable 
import torch 
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
import os

try:
    os.mkdir('results')
except:
    pass

batchSize = 64 
imageSize = 64 


transform = transforms.Compose([transforms.Scale(imageSize),
transforms.ToTensor(),
transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = dsets.MNIST(root='./data',download=True,transform=transform)
dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=64,shuffle=True)

def initialiseWeights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0,.02)
        model.bias.data.fill_(0)


generatorNN = Generator()
generatorNN.apply(initialiseWeights)
generatorNN.cuda()


discriminatorNN = Discriminator()
discriminatorNN.apply(initialiseWeights)
discriminatorNN.cuda()

criterion = nn.BCELoss()
disOptimizer = optim.Adam(discriminatorNN.parameters(),lr=.0002,betas=(0.5,.999))
genOptimizer = optim.Adam(generatorNN.parameters(),lr=.0002,betas=(0.5,.999))

for epoch in range(20):

    for i, data in enumerate(dataloader):

        discriminatorNN.zero_grad()

        realD,_ = data
        real= Variable(realD).cuda()
        target = torch.ones(real.size()[0]) #Size of minibatch
        target = Variable(target).cuda()

        output = discriminatorNN(real)
        discriminatorError_R = criterion(output,target)

        #NN of generator takes input a vector of size 100

        noise = Variable(torch.randn
        (
            real.size()[0], # batch_size
                100, #generate 100 values
                1,1  # generated 100 values have dimension 1x1
        )).cuda()
        fake = generatorNN(noise)
        target = Variable(torch.zeros(real.size()[0])).cuda()
        #fake will not be used in backpropagation
        #hence detaching the gradient to save some memory
        output = discriminatorNN(fake.detach())  
        discriminatorError_F = criterion(output,target)

        totalDiscriminatorError = discriminatorError_F + discriminatorError_R
        totalDiscriminatorError.backward()
        disOptimizer.step()


        #GENERATOR
        generatorNN.zero_grad()
        target=Variable(torch.ones(real.size()[0])).cuda()
        output = discriminatorNN(fake)
        generatorError = criterion(output,target)
        generatorError.backward()
        genOptimizer.step()
    
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'%(epoch+1,20,i,len(dataloader),totalDiscriminatorError.data[0],generatorError.data[0]))

        if i % 100 ==0:
            vutils.save_image(realD,'%s/real_samples.png'%('./results'),normalize=True)
            fake=generatorNN(noise)
            vutils.save_image(fake.data,'%s/fake_samples_epoch_%03d.png'%('./results',epoch),normalize=True)
