import torch.nn as nn
import torch.nn.functional as F

#定义resnet的核心模块resBlock
class resBlock(nn.Module):
    def __init__(self,in_channel):
        super(resBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel,in_channel,3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        ]
        # 将算子 串联成一个卷积核
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self,x):
        return x + self.conv_block(x);

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        net = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        #downsample
        in_channel = 64
        out_channel = in_channel * 2
        for _ in range(2):
            net += [
                nn.Conv2d(in_channel,out_channel,3, 2, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ]
            in_channel = out_channel
            out_channel = in_channel * 2

        for _ in range(9):
            net +=[resBlock(in_channel)]

        #unsampleing
        out_channel = in_channel //2
        for _ in range(2):
            net += [nn.ConvTranspose2d(in_channel,out_channel,3,
                                       stride=2,padding=1,
                                       output_padding=1),
                    nn.InstanceNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                    ]

            in_channel = out_channel
            out_channel = in_channel // 2

        net += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,3, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*net)

    def forward(self,x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        model = []