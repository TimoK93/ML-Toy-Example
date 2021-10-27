import torch as t
from torch_scatter import scatter_max

assert t.cuda.is_available()

print("Device:", t.cuda.current_device())
print("Capability:", t.cuda.get_device_capability(t.cuda.current_device()))
print("Properties:", t.cuda.get_device_properties(t.cuda.current_device()))
print("Architecture list:", t.cuda.get_arch_list())
print("\n")

device = "cuda:" + str(t.cuda.current_device())

t.cuda.get_device_name()

src = t.Tensor([
    [[1, 2], [0, 0], [0, 0]],
    [[0, 0], [3, 4], [0, 0]],
    [[0, 0], [0, 0], [5, 6]],
]).to(device)

index = t.tensor([
    [[0, 0], [1, 0], [2, 0]],
    [[0, 1], [1, 1], [2, 1]],
    [[0, 2], [1, 2], [2, 2]],
]).to(device)

sh = src.shape
src = src.reshape((sh[0] * sh[1], sh[2]))
index = (index[:, :, 0] + index.shape[0] * index[:, :, 1]).flatten()

out = t.zeros_like(src)

out, argmax = scatter_max(src, index, out=out, dim=0)
out = out.reshape(sh)
print(out)

