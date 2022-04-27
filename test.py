
from rivagan import RivaGAN
import pdb
import numpy as np
import torch

def get_acc(y_true, y_pred):
    # assert y_true.size() == y_pred.size()
    return (torch.Tensor(y_pred) >= 0.0).eq(torch.Tensor(y_true) >= 0.5).sum().float().item() / torch.Tensor(y_pred).numel()

#data = tuple([1]*32)
data = tuple([1,0,0,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0])
model = RivaGAN.load('./model_critic_adversary.pt')
model.encode('./mov测试文件.mov', data, './mov测试文件_critic_adversary.avi')
acc = []
for recovered_data in model.decode('./mov测试文件_critic_adversary.avi'):
    acc.append(get_acc(data, recovered_data))
    print('now acc:%s'%str(acc[-1]))
    print('now averate acc:%s'%str(np.mean(np.array(acc))))

print(np.mean(np.array(acc)))
