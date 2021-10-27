from torchvision import models
import torch as t
from torch_scatter import scatter_max

assert t.cuda.is_available()

print("Device:", t.cuda.current_device())
print("Capability:", t.cuda.get_device_capability(t.cuda.current_device()))
print("Properties:", t.cuda.get_device_properties(t.cuda.current_device()))
print("Architecture list:", t.cuda.get_arch_list())
print("\n")

device = "cuda:" + str(t.cuda.current_device())

pretrained_model = models.resnet50(pretrained=True,).to(device)

src = t.zeros((1, 3, 1000, 1000)).to(device)

out = pretrained_model(src)
print(out)

