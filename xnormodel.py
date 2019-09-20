import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import act_fn, print_values
from torch.autograd import Function,Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""*************    XNOR-Net Basic Blocks    ****************"""

class BinActiv(Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension
    '''
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input #tensor.Forward should has only one output, or there will be another grad
    
    @classmethod
    def Mean(cls,input):
        return torch.mean(input.abs(),1,keepdim=True) #the shape of mnist data is (N,C,W,H)

    @staticmethod
    def backward(ctx,grad_output): #grad_output is a Variable
        input,=ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input #Variable

BinActive = BinActiv.apply

class BinConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,drop_ratio=0,groups=1,bias=False):
        super(BinConv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.drop_ratio = drop_ratio #################
        self.groups = groups
        self.bias = bias
        self.layer_type = 'BinConv2d'

        self.bn = nn.BatchNorm2d(in_channels,eps=1e-4,momentum=0.1,affine=True)
        if self.drop_ratio != 0:########
            self.drop = nn.Dropout(self.drop_ratio)#############
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,
                            groups=groups,bias=bias)
        self.relu = nn.ReLU()

    def forward(self,x):
        #block structure is BatchNorm -> BinActiv -> BinConv -> Relu
        x = self.bn(x)
        A = BinActiv().Mean(x)
        x = BinActive(x)
        if self.drop_ratio != 0:#################
            x = self.drop(x)#####################
        k = torch.ones(1,1,self.kernel_size,self.kernel_size).mul(1/(self.kernel_size**2)) #out_channels and in_channels are both 1.constrain kernel as square
        k = Variable(k.cuda())
        K = F.conv2d(A,k,bias=None,stride=self.stride,padding=self.padding,dilation=self.dilation)
        x = self.conv(x)
        x = torch.mul(x,K)
        x = self.relu(x)
        return x

class BinLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(BinLinear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = nn.BatchNorm1d(in_features,eps=1e-4,momentum=0.1,affine=True)
        self.linear = nn.Linear(in_features,out_features,bias=False)

    def forward(self,x):
        x = self.bn(x)
        beta = BinActiv().Mean(x).expand_as(x)
        x = BinActive(x)
        x = torch.mul(x,beta)
        x = self.linear(x)
        return x

"""************* Original PNN Implementation ****************"""

class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.level = level
        self.layers = nn.Sequential(
            #nn.ReLU(True),
            #nn.BatchNorm2d(in_planes),  #TODO paper does not use it!
            #nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            BinConv2d(in_planes, out_planes, kernel_size=1, stride=1), # instead of conv; try batchnorm after binconv if working bad, i think
			nn.ReLU(True),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()   #fill with uniform noise
            #self.noise = (2 * self.noise - 1) * self.level
            self.noise = nn.Parameter((2 * self.noise - 1) * self.level, requires_grad=False).to(device)
        y = torch.add(x, self.noise)
        return self.layers(y)   #input, perturb, relu, batchnorm, conv1x1


class NoiseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, planes, level),  #perturb, relu, conv1x1
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),  #TODO paper does not use it!
            #nn.BatchNorm2d(planes), # moved to here instead of after the noise layer
            NoiseLayer(planes, planes, level),  #perturb, relu, conv1x1
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class NoiseResNet(nn.Module):
    def __init__(self, block, nblocks, nfilters, nclasses, pool, level, first_filter_size=3):
        super(NoiseResNet, self).__init__()
        self.in_planes = nfilters
        if first_filter_size == 7:
            pool = 1
            self.pre_layers = nn.Sequential(
                # try binconv; TODO: batchnorm before or replace completely
                nn.Conv2d(3, nfilters, kernel_size=first_filter_size, stride=2, padding=3, bias=False),
                #BinConv2d(3, nfilters, kernel_size=first_filter_size, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(nfilters),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )
        elif first_filter_size == 3:
            pool = 4
            self.pre_layers = nn.Sequential(
                # try binconv; TODO: batchnorm before or replace completely
                #BinConv2d(3, nfilters, kernel_size=first_filter_size, stride=1, padding=1, bias=False),
                nn.Conv2d(3, nfilters, kernel_size=first_filter_size, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(nfilters),
                nn.ReLU(True),
            )
        elif first_filter_size == 0:
            print('\n\nThe original noiseresnet18 model does not support noise masks in the first layer, '
                  'use perturb_resnet18 model, or set first_filter_size to 3 or 7\n\n')
            return

        #self.pre_layers[0].weight.requires_grad = False # (5) Felix added this, first layer rand conv
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], stride=1, level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level, drop_ratio=0.5)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        # try binary linear layer; replace if accuracy is low
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)
        #self.linear = BinLinear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, filter_size=1, drop_ratio=0):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                # try binconv; TODO: batchnorm before or replace completely
                #nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BinConv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, drop_ratio=drop_ratio),
                #nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8



def noiseresnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=7,
                  pool_type=None, input_size=None, scale_noise=1, act='relu', use_act=True, dropout=0.5, unique_masks=False,
                  debug=False, noise_type='uniform', train_masks=False, mix_maps=None):
    return NoiseResNet(NoiseBasicBlock, [2, 2, 2, 2], nfilters=nfilters, pool=avgpool, nclasses=nclasses,
                       level=level, first_filter_size=first_filter_size)
