import os
import copy
import numpy as np
import math
import torch
import torch.nn as nn

def readtextfile(filename):
	with open(filename) as f:
		content = f.readlines()
	f.close()
	return content

def writetextfile(data, filename):
	with open(filename, 'w') as f:
		f.writelines(data)
	f.close()

def delete_file(filename):
	if os.path.isfile(filename) == True:
		os.remove(filename)

def eformat(f, prec, exp_digits):
	s = "%.*e"%(prec, f)
	mantissa, exp = s.split('e')
	# add 1 to digits as 1 is taken by sign +/-
	return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))

def saveargs(args):
	path = args.logs
	if os.path.isdir(path) == False:
		os.makedirs(path)
	with open(os.path.join(path,'args.txt'), 'w') as f:
		for arg in vars(args):
			f.write(arg+' '+str(getattr(args,arg))+'\n')

def init_params(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal(m.weight, mode='fan_out')
			if m.bias:
				nn.init.constant(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant(m.weight, 1)
			nn.init.constant(m.bias, 0)
		elif isinstance(m, nn.Linear):
			nn.init.normal(m.weight, std=1e-3)
			if m.bias:
				nn.init.constant(m.bias, 0)

def weights_init(m):
	if isinstance(m, nn.Conv2d):
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
	elif isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()


class Counter:  #not used currently
	def __init__(self):
		self.mask_size = 0

	def update(self, size):
		self.mask_size += size

	def get_total(self):
		return self.mask_size


def act_fn(act):
	if act == 'relu':
		act_ = nn.ReLU(inplace=False)
	elif act == 'lrelu':
		act_ = nn.LeakyReLU(inplace=True)
	elif act == 'prelu':
		act_ = nn.PReLU()
	elif act == 'rrelu':
		act_ = nn.RReLU(inplace=True)
	elif act == 'elu':
		act_ = nn.ELU(inplace=True)
	elif act == 'selu':
		act_ = nn.SELU(inplace=True)
	elif act == 'tanh':
		act_ = nn.Tanh()
	elif act == 'sigmoid':
		act_ = nn.Sigmoid()
	else:
		print('\n\nActivation function {} is not supported/understood\n\n'.format(act))
		act_ = None
	return act_


def print_values(x, noise, y, unique_masks, n=2):
	np.set_printoptions(precision=5, linewidth=200, threshold=1000000, suppress=True)
	print('\nimage: {}  image0, channel0          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[0, 0, 0, 0, :n].cpu().numpy()))
	print('image: {}  image0, channel1          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[0, 1, 0, 0, :n].cpu().numpy()))
	print('\nimage: {}  image1, channel0          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[1, 0, 0, 0, :n].cpu().numpy()))
	print('image: {}  image1, channel1          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[1, 1, 0, 0, :n].cpu().numpy()))
	if noise is not None:
		print('\nnoise {}  channel0, mask0:           {}'.format(list(noise.size()), noise.data[0, 0, 0, 0, :n].cpu().numpy()))
		print('noise {}  channel0, mask1:           {}'.format(list(noise.size()), noise.data[0, 0, 1, 0, :n].cpu().numpy()))
		if unique_masks:
			print('\nnoise {}  channel1, mask0:           {}'.format(list(noise.size()), noise.data[0, 1, 0, 0, :n].cpu().numpy()))
			print('noise {}  channel1, mask1:           {}'.format(list(noise.size()), noise.data[0, 1, 1, 0, :n].cpu().numpy()))

	print('\nmasks: {} image0, channel0, mask0:  {}'.format(list(y.size()), y.data[0, 0, 0, 0, :n].cpu().numpy()))
	print('masks: {} image0, channel0, mask1:  {}'.format(list(y.size()), y.data[0, 0, 1, 0, :n].cpu().numpy()))
	print('masks: {} image0, channel1, mask0:  {}'.format(list(y.size()), y.data[0, 1, 0, 0, :n].cpu().numpy()))
	print('masks: {} image0, channel1, mask1:  {}'.format(list(y.size()), y.data[0, 1, 1, 0, :n].cpu().numpy()))
	print('\nmasks: {} image1, channel0, mask0:  {}'.format(list(y.size()), y.data[1, 0, 0, 0, :n].cpu().numpy()))
	print('masks: {} image1, channel0, mask1:  {}'.format(list(y.size()), y.data[1, 0, 1, 0, :n].cpu().numpy()))
	print('masks: {} image1, channel1, mask0:  {}'.format(list(y.size()), y.data[1, 1, 0, 0, :n].cpu().numpy()))
	print('masks: {} image1, channel1, mask1:  {}'.format(list(y.size()), y.data[1, 1, 1, 0, :n].cpu().numpy()))

"""********** Binary operations class **********"""

class Binop:
	def __init__(self,model):
		count_targets = 0
		for m in model.modules():
			if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
				count_targets += 1
		self.bin_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
		self.num_of_params = len(self.bin_range)
		self.saved_params = []
		self.target_modules = []
		for m in model.modules():
			print(m)
			if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
				tmp = m.weight.data.clone()
				self.saved_params.append(tmp) #tensor
				self.target_modules.append(m.weight) #Parameter
	
	def ClampWeights(self):
		for index in range(self.num_of_params):
			self.target_modules[index].data = self.target_modules[index].data.clamp(-1.0,1.0)
			#self.target_modules[index].data.clamp_(-1.0,1.0) # x.clamp_ should be inplace????
	
	def SaveWeights(self):
		for index in range(self.num_of_params):
			self.saved_params[index].copy_(self.target_modules[index].data)

	def BinarizeWeights(self):
		for index in range(self.num_of_params):
			n = self.target_modules[index].data[0].nelement()
			s = self.target_modules[index].data.size()
			if len(s) == 4:
				alpha = self.target_modules[index].data.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n)
			elif len(s) == 2:
				alpha = self.target_modules[index].data.norm(1,1,keepdim=True).div(n)
			self.target_modules[index].data = self.target_modules[index].data.sign().mul(alpha.expand(s))
	
	def Binarization(self):
		self.ClampWeights()
		self.SaveWeights()
		self.BinarizeWeights()
	
	def Restore(self):
		for index in range(self.num_of_params):
			self.target_modules[index].data.copy_(self.saved_params[index])
	
	def UpdateBinaryGradWeight(self):
		for index in range(self.num_of_params):
			if hasattr(self.target_modules[index].grad, 'data'):
				weight = self.target_modules[index].data
				n = weight[0].nelement()
				s = weight.size()
				if len(s) == 4:
					alpha = weight.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand(s)
				elif len(s) == 2:
					alpha = weight.norm(1,1,keepdim=True).div(n).expand(s)
				alpha[weight.le(-1.0)] = 0
				alpha[weight.ge(1.0)] = 0
				alpha = alpha.mul(self.target_modules[index].grad.data)
				add = weight.sign().mul(self.target_modules[index].grad.data)
				if len(s) == 4:
					add = add.sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand(s)
				elif len(s) == 2:
					add = add.sum(1,keepdim=True).div(n).expand(s)
				add = add.mul(weight.sign())
				self.target_modules[index].grad.data = alpha.add(add)