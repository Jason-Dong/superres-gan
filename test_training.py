#from reds_dataset import Reds
from data.vimeo_dataset import VimeoDataset
from models.superresblock import SuperRes
from torch.optim import Adam
from torch import nn
from models.lossfunc import PerceptualLoss
from train import train, test, setup_data
from torchvision.models import vgg16
import cv2
import torch

path_to_data = "./"
use_cuda = False

super_res_data = VimeoDataset(path_to_data)

model = SuperRes()
if use_cuda:
    model.cuda()


optimizer = Adam(model.parameters(), lr=1e-3) #parameters returns an iterator over the models parameters

#criterion = PerceptualLoss(loss_network=relu2_2)

ftrain = "./"#file path to train
ftest = "./"#file path to test

trainloader_task_2, testloader_task_2 = setup_data(ftrain, ftest, 1,0)

epochs = 1 #to see if we have no errors, we can set this value to one

for epoch in range(epochs):
    print("Current Epoch: %d" % epoch)
    train(model, trainloader_task_2, optimizer, cuda_enabled=use_cuda)
    test(model, testloader_task_2, cuda_enabled=use_cuda)
    filename = 'model' +str(epoch) + '.pt'
    torch.save(model, filename)

# path_to_test_img = "./output.jpg"
# x = cv2.imread(path_to_test_img, cv2.IMREAD_COLOR)
# x = super_res_data.convert_image_small(x)
# print(x)
# x = x.unsqueeze(0)
# output_test = model(x)

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torchvision.transforms as transforms
# print(output_test.shape)
# print(x.shape)
# for i in output_test:
#     print(i.shape)
#     plt.imshow(transforms.ToPILImage()(i))


#imgplot = plt.imshow(output_test.detach.numpy())
#plt.imshow(torchvision.transforms.ToPILImage()(output_test), interpolation="bicubic")
#temp = torch.flatten(output_test, start_dim = 2)
#print(temp.shape)
#plt.imshow(output_test.permute(1,2,0))
#plt.imshow(output_test.detach().numpy()[0])

# output_test = model(prep_img(super_res_data[0][0]))
