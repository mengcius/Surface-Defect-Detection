import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.constant_(m.weight.data, 0.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

class InceptionA(nn.Module):
    
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = nn.Conv2d
        self.branch1x1 = conv_block(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 24, kernel_size=1)
        self.branch5x5_2 = conv_block(24, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(32, 32, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(32, 32, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

    
class SegmentNet(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super(SegmentNet, self).__init__()

#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channels, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)  
                        )
        
        self.Mixed_1b = InceptionA(32, pool_features=2)
#         self.Mixed_1c = InceptionA(82, pool_features=2)
        
        self.layer3 = nn.Sequential(
                            nn.Conv2d(82, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)
                        )        
        
        self.layer4 = nn.Sequential(
                            nn.Conv2d(64, 1024, 15, stride=1, padding=7),  # 64,1024
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True)
                        )

        self.layer5 = nn.Sequential(
                            nn.Conv2d(1024, 1, 1),
                            nn.ReLU(inplace=True)
                        )

        if init_weights == True:
            pass

    def forward(self, x):
        x = self.layer1(x)
#         x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x1 = self.Mixed_1b(x)
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2, padding=1)      
#         print('x1:', x1.shape)
        
#         x2 = self.Mixed_1c(x1)
#         x2 = F.max_pool2d(x2, kernel_size=5, stride=2, padding=2)
#         print('x2:', x2.shape)

#         x3 = self.Mixed_1d(x2)
#         x3 = F.max_pool2d(x3, kernel_size=5, stride=2, padding=2)
#         x3 = self.layers(x) # torch.Size([4, 64, 88, 32]), torch.Size([4, 256, 88, 32])
#         print('x3:', x3.shape)

        x3 = self.layer3(x1)
        x4 = self.layer4(x3) # torch.Size([4, 1024, 88, 32]) 
        x5 = self.layer5(x4) # torch.Size([4, 1, 88, 32])
#         print('x:', x4.shape, x5.shape) 

        return {"f":x4, "seg":x5}


class DecisionNet(nn.Module):
    
    def __init__(self, init_weights=True):
        super(DecisionNet, self).__init__()

        self.layer1 = nn.Sequential(
                            # nn.MaxPool2d(2),
            
#                             nn.Conv2d(1025, 8, 5, stride=1, padding=2),
#                             nn.BatchNorm2d(8),
#                             nn.ReLU(inplace=True),
#                             nn.MaxPool2d(2),
            
                            # Depthwise+Pointwise
                            nn.Conv2d(1025, 1025, kernel_size=5, stride=1, padding=2, groups=1025, bias=False),
                            nn.BatchNorm2d(1025),
                            nn.Conv2d(1025, 8, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(8),
                            nn.MaxPool2d(2),
            
#                             nn.Conv2d(8, 16, 5, stride=1, padding=2),
#                             nn.BatchNorm2d(16),
#                             nn.ReLU(inplace=True),
# #                             nn.MaxPool2d(2),
            
                            # Depthwise+Pointwise
                            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2, groups=8, bias=False),
                            nn.BatchNorm2d(8),
                            nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(16),
                            nn.MaxPool2d(2),            
            
                            nn.Conv2d(16, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True)
                        )

        self.fc =  nn.Sequential(
                            nn.Linear(66, 1, bias=False),
                            nn.Sigmoid()
                        )

        if init_weights == True:
            pass

    def forward(self, f, s):
        xx = torch.cat((f, s), 1)
        x1 = self.layer1(xx)
        x2 = x1.view(x1.size(0), x1.size(1), -1)
        s2 = s.view(s.size(0), s.size(1), -1)

        x_max, x_max_idx = torch.max(x2, dim=2)
        x_avg = torch.mean(x2, dim=2)
        s_max, s_max_idx = torch.max(s2, dim=2)
        s_avg = torch.mean(s2, dim=2)

        y = torch.cat((x_max, x_avg, s_avg, s_max), 1)
        y = y.view(y.size(0), -1)

        return self.fc(y)


if  __name__=='__main__':
    
    snet = SegmentNet()
    dnet = DecisionNet() 
    img  =  torch.randn(4, 3, 704, 256)

    snet.eval()
    snet = snet.cuda()
    dnet = dnet.cuda()
    img = img.cuda()

    ret = snet(img)
    f = ret["f"]
    s = ret["seg"]

    c = dnet(f, s)
    print(c)
    pass

