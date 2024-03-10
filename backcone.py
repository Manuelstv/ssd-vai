from model import VGGBase
import torch

image = torch.zeros([1,3,300,300])
model = VGGBase()

model(image)