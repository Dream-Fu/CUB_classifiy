import torch as t
import torch.nn as nn
import torchvision.models as m

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.features = m.vgg16(pretrained=True).features
        # 去除最后的fc
        self.features = nn.Sequential(*list(self.features.children())[:-1])
        # torch.Size([1, 512, 16, 16])
        self.fc = nn.Linear(512 * 512, 200)

        for param in self.features.parameters():
            param.requires_grad = True
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size()[0]
        out = self.features(x)
        # torch.Size([1, 512, 16, 16])
        out = out.view(N, 512, 16*16)
        # N,512,16*16 * N,16*16, 512
        out = t.bmm(out, out.transpose(1, 2)) / (16 * 16)
        # N, 512,512
        out = out.view(N, 512*512)
        out = t.sqrt(out + 1e-5)
        out = t.nn.functional.normalize(out)
        out = self.fc(out)
        return out





if __name__ == '__main__':
    model = vgg16()
    input = t.randn((1, 3, 256, 256))
    # to onnx
    # input_names = ['input']
    # output_names = ['output']
    # t.onnx.export(model, input, './BCNN.onnx',
    #               input_names=input_names, output_names=output_names,
    #               verbose=True, opset_version=11)
    out = model(input)
    print(out.shape)


