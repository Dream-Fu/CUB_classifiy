import torch.nn as nn
import torchvision.models as m
import torch as t

class BCNN(nn.Module):
    def __init__(self):
        super(BCNN, self).__init__()
        resnet50 = m.resnet50(pretrained=True)
        self.features = nn.Sequential(resnet50.conv1,
                                      resnet50.bn1,
                                      resnet50.relu,
                                      resnet50.maxpool,
                                      resnet50.layer1,
                                      resnet50.layer2,
                                      resnet50.layer3,
                                      resnet50.layer4)
        self.fc = nn.Sequential(nn.Linear(2048*2048, 200))

    def forward(self, x):
        out = self.features(x)
        N = out.size()[0]
        Y = out.size(2)
        # [1, 2048, 8* 8]
        out = out.view((N, out.size(1), Y**2))
        # [1, 2048, 2048]
        out = t.bmm(out, out.transpose(1, 2)) // Y**2
        # [1, 2048*2048]
        out = out.view(N, -1)
        # x = torch.nn.functional.normalize(torch.sign(x) *
        #                                   torch.sqrt(torch.abs(x) + 1e-10))
        out = t.nn.functional.normalize(out)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    # pre_dict = m.resnet50(pretrained=True)
    # print(pre_dict)
    model = BCNN()
    input = t.randn((1, 3, 256, 256))
    out = model(input)
    print(out.shape)