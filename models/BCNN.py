import torch.nn as nn
import torchvision.models as m
import torch as t
from torchsummary import summary

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
        self.fc = nn.Linear(2048*2048, 200)
        # for param in self.features.parameters():
        #     param.requires_grad = True
        # t.nn.init.kaiming_normal_(self.fc.weight.data)
        # if self.fc.bias is not None:
        #     t.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        out = self.features(x)
        N = out.size()[0]
        Y = out.size(2)
        # [1, 2048, 8* 8]
        out = out.view((N, out.size(1), Y**2))
        # [1, 2048, 2048]
        out = t.bmm(out, t.transpose(out, 1, 2)) / Y**2
        # [1, 2048*2048]
        out = out.view(N, -1)
        out = t.nn.functional.normalize(t.sign(out) * t.sqrt(t.abs(out) + 1e-10))
        # out = t.nn.functional.normalize(out)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    # pre_dict = m.vgg16(pretrained=True)
    # print(pre_dict)
    model = BCNN().cuda()
    input = t.randn((1, 3, 256, 256))
    # out = model(input)
    # print(out.shape)
    summary(model, (3, 256, 256))