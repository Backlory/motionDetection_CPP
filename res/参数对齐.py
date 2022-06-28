import torch
import io


#a = torch.load("sintel-weight-check2.pt").state_dict()
a = torch.load("mdhead-weight-cpp.pt").state_dict()
b = torch.load("model_Train_MDHead_and_save_bs8_60.pkl")['state_dict']

aa = list(a.keys())
bb = list(b.keys())
aa.sort()
bb.sort()
assert(len(aa) == len(bb))
for idx in range(len(aa)):
    kb = bb[idx]
    ka = aa[idx]
    if True:
        print(idx, ka," cpp-- ",a[ka].shape)
        print(idx, kb[5:]," python-- ",b[kb].shape)
        print(torch.sum(b[kb]-a[ka]))
        print("")


