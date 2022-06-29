import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
#from epdn.OmniDepth_network import ConELUBlock
import numpy as np
from util.adain import AdaIN
import models.fusion_networks as FN
import torch.nn.functional as F
from models.ffm import MixFFM as FSM
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			return lr_l
		
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)
	
	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert (torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)
	init_weights(net, init_type)
	return net


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)


def define_Encoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_noada':
		netG = ResnetEncoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'resnet_6blocks':
		netG = ResnetEncoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet':
		netG = UnetEncoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_noada':
		netG = UnetEncoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy':
		netG = HieUnetEncoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy_sf':
		netG = HieSFUnetEncoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

def define_Encoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_noada':
		netG = ResnetEncoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'resnet_6blocks':
		netG = ResnetEncoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet':
		netG = UnetEncoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_hierarchy':
		netG = HieUnetEncoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy':
		netG = HieUnetEncoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy_sf':
		netG = HieSFUnetEncoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

def define_Decoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_noada':
		netG = ResnetNoadaDecoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'resnet_6blocks':
		netG = ResnetDecoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'resnet_add_6blocks':
		netG = ResnetAddDecoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet':
		netG = UnetDecoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_hierarchy':
		netG = HieUnetDecoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy':
		netG = HieUnetAddDecoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy_sf':
		netG = HieSFUnetAddDecoder_H(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

def define_Decoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_noada':
		netG = ResnetNoadaDecoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'resnet_6blocks':
		netG = ResnetDecoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'resnet_add_6blocks':
		netG = ResnetAddDecoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet':
		netG = UnetDecoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_hierarchy':
		netG = HieUnetDecoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy':
		netG = HieUnetAddDecoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	elif which_model_netG == 'unet_add_hierarchy_sf':
		netG = HieSFUnetAddDecoder_C(input_nc, output_nc, ngf, n_width, n_height, n_blocks, norm_layer=norm_layer, use_dropout=use_dropout)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=4, norm='instance', use_sigmoid=False, init_type='normal', gpu_ids=[],use_feature=False):
	netD = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netD == 'basic':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,use_feature=use_feature)
	elif which_model_netD == 'n_layers':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,use_feature=use_feature)
	elif which_model_netD == 'pixel':
		netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' %
		                          which_model_netD)
	return init_net(netD, init_type, gpu_ids)



# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()
	
	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(input)
	
	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)

#=====================================================


class SpecificFeature(nn.Module):
    def __init__(self, n_channels_in, n_channels_out,  reduction_ratio):
        super(SpecificFeature, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in/ float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_out)
        )


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        max_pool = F.max_pool2d(x, kernel)

        
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)
        

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool#.repeat(1,1,kernel[0], kernel[1])
        return out

# Seprate Content and Style #

class StyleEncoder(nn.Module):
	def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
		super(NoiseEncoder, self).__init__()
		self.model = []
		self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
		for _ in range(2):
			self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
			dim *= 2
		for i in range(n_downsample - 2):
			self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
		#self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
		self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
		self.model = nn.Sequential(*self.model)
		self.output_dim = dim

	def forward(self, x):
		return self.model(x)

class ContentEncoder(nn.Module):
	def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
		super(ContentEncoder, self).__init__()
		self.model = []
		self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
		# downsampling blocks
		for _ in range(n_downsample):
			self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
			dim *= 2
		# residual blocks
		self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
		self.model = nn.Sequential(*self.model)
		self.output_dim = dim

	def forward(self, x):
		return self.model(x)


#=======================================================

#####################
## Encoder-Decoder ##
#####################

class ResnetEncoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(ResnetEncoder, self).__init__()
		self.n_width = n_width
		self.n_height = n_height
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		activation = nn.ReLU(True)
		
		### feature extractor
		feture_extract = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		#feture_extract += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, ngf*2, kernel_size=5, padding=0), norm_layer(ngf*2), activation]
		self.feture_extract = nn.Sequential(*feture_extract)
		# encoder
		mult = 1
		sub_n_blocks = 1
		generator_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		for i in range(sub_n_blocks):
			generator_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_e0 = nn.Sequential(*generator_e0)

		mult = 2
		self.mult = mult
		generator_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		for i in range(sub_n_blocks):
			generator_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_e1 = nn.Sequential(*generator_e1)

		#mult = 8
		#self.mult = mult
		#generator_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		#generator_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_e2 = nn.Sequential(*generator_e2)

		self.generator_s = nn.Sequential(nn.AvgPool2d(2, stride=2), 
			nn.Conv2d(ngf * mult * 2 , ngf * mult * 2 , kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2 ), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf * mult * 2 , ngf * mult * 2 , kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2 ), activation
			)

		### resnet blocks

		generator_c = []
		for i in range(n_blocks):
			generator_c += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
			self.generator_c = nn.Sequential(*generator_c)

		#self.specific_feature = nn.Sequential(
		#	nn.Linear((self.ngf * self.mult * 2) * (self.n_height // (self.mult*2)) * (self.n_width // (self.mult*2)), ngf *  self.mult * 2),
		#	nn.LeakyReLU(0.2, True),
		#	nn.Linear(ngf *  self.mult * 2, ngf *  self.mult * 2),
		#	)
		self.specific_feature = SpecificFeature(ngf * mult * 2, ngf * mult * 2, 8)

	def forward(self, input):
		f = self.feture_extract(input)
		e0 = self.generator_e0(f)
		e1 = self.generator_e1(e0)
		#e2 = self.generator_e2(e1)
		c = self.generator_c(e1)
		s = self.generator_s(e1)
		#print('s', s.shape)

		#line = s.view(-1, (self.ngf * self.mult * 2) * (self.n_height // (self.mult*2)) * (self.n_width // (self.mult*2)))
		s = self.specific_feature(s)
		#s = s.unsqueeze(2)
		#s = s.unsqueeze(3)

		return [f, e0, e1], c, s


class ResnetDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(ResnetDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self.ada = AdaIN()
		activation = nn.ReLU(True)
		### decoder
		mult = 4
		generator_t = []
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
			self.generator_t = nn.Sequential(*generator_t)

		#mult = 16
		#generator_d2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 4
		sub_n_blocks = 1
		generator_d1 = []#[nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		generator_d1 += [nn.ConvTranspose2d(int(ngf * mult * 2), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d1 += [nn.Conv2d(int(ngf * mult / 2) , int(ngf * mult / 2) , kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			#print('do rui feng')
			generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 2
		generator_d0 = []#[nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		generator_d0 += [nn.ConvTranspose2d(int(ngf * mult* 2), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d0 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d0 = nn.Sequential(*generator_d0)

		conv_skip_1 = [nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), norm_layer(ngf*4), activation]
		conv_skip_0 = [nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), norm_layer(ngf*2), activation]
		conv_skip_f = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		self.conv_skip_1 = nn.Sequential(*conv_skip_1)
		self.conv_skip_0 = nn.Sequential(*conv_skip_0)
		self.conv_skip_f = nn.Sequential(*conv_skip_f)

		final = []
		final += [nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, f, c, s):
		#print('f', f[0].shape,'f1', f[1].shape)
		t = self.ada(c, s)
		dt = self.generator_t(t)
		skip = self.conv_skip_1(f[2])
		cat = torch.cat((dt, skip), 1)
		d1 = self.generator_d1(cat)

		skip = self.conv_skip_0(f[1])
		cat = torch.cat((d1, skip), 1)
		d0 = self.generator_d0(cat)

		skip = self.conv_skip_f(f[0])
		cat = torch.cat((d0, skip), 1)
		output = self.final(cat)

		return output

class ResnetAddDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(ResnetAddDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self.ada = AdaIN()
		activation = nn.ReLU(True)
		### decoder
		mult = 4
		generator_t = []
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
			self.generator_t = nn.Sequential(*generator_t)

		#mult = 16
		#generator_d2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 4
		sub_n_blocks = 1
		generator_d1 = []#[nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		generator_d1 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d1 += [nn.Conv2d(int(ngf * mult / 2) , int(ngf * mult / 2) , kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			#print('do rui feng')
			generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 2
		generator_d0 = []#[nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		generator_d0 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d0 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d0 = nn.Sequential(*generator_d0)

		#conv_skip_1 = [nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), norm_layer(ngf*4), activation]
		#conv_skip_0 = [nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), norm_layer(ngf*2), activation]
		#conv_skip_f = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		#self.conv_skip_1 = nn.Sequential(*conv_skip_1)
		#self.conv_skip_0 = nn.Sequential(*conv_skip_0)
		#self.conv_skip_f = nn.Sequential(*conv_skip_f)

		final = []
		final += [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, f, c, s):
		#print('f', f[0].shape,'f1', f[1].shape)
		t = self.ada(c, s)
		dt = self.generator_t(t)
		#skip = self.conv_skip_1(f[2])
		cat = dt + f[2]
		d1 = self.generator_d1(cat)

		#skip = self.conv_skip_0(f[1])
		cat = d1 + f[1]
		d0 = self.generator_d0(cat)

		#skip = self.conv_skip_f(f[0])
		cat = d0 + f[0]
		output = self.final(cat)

		return output


class Decoder(nn.Module):
	def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
		super(Decoder, self).__init__()

		self.model = []
		# AdaIN residual blocks
		self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
		# upsampling blocks
		for _ in range(n_upsample):
			self.model += [nn.Upsample(scale_factor=2),
			               Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
		dim //= 2
		# use reflection padding in the last conv layer
		self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)] # tanh
		self.model = nn.Sequential(*self.model)

	def forward(self, x):
		return self.model(x)

###########
## U-Net ##
###########

class UnetEncoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(UnetEncoder, self).__init__()
		self.n_width = n_width
		self.n_height = n_height
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		activation = nn.ReLU(True)
		
		### feature extractor
		feture_extract = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		feture_extract += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, ngf*2, kernel_size=5, padding=0), norm_layer(ngf*2), activation]
		self.feture_extract = nn.Sequential(*feture_extract)

		# encoder
		mult = 2
		generator_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		generator_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_e0 = nn.Sequential(*generator_e0)

		mult = 4
		generator_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		generator_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_e1 = nn.Sequential(*generator_e1)

		mult = 8
		self.mult = mult
		generator_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		generator_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_e2 = nn.Sequential(*generator_e2)

		### resnet blocks
		generator_t = []
		mult = 2**(3 +1)
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout)]
			self.generator_t = nn.Sequential(*generator_t)

		
		self.generator_s = nn.Sequential(nn.Conv2d(ngf * mult , ngf * mult , kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult ), activation,
			nn.Conv2d(ngf * mult , ngf * mult , kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult ), activation)

		self.specific_feature = nn.Sequential(
			nn.Linear((self.ngf * self.mult * 2) * (self.n_height // (self.mult * 4)) * (self.n_width // (self.mult * 4)), ngf * self.mult* 2),
			nn.LeakyReLU(0.2, True),
			nn.Linear(ngf * self.mult* 2, ngf * self.mult* 2),
			)

	def forward(self, input):
		f = self.feture_extract(input)
		e0 = self.generator_e0(f)
		e1 = self.generator_e1(e0)
		e2 = self.generator_e2(e1)
		c = self.generator_t(e2)
		s = self.generator_s(e2)
		line = s.view(-1, (self.ngf * self.mult * 2) * (self.n_height // (self.mult * 4)) * (self.n_width // (self.mult * 4)))
		#print('self.n_height // (self.mult * 2)', self.n_height // (self.mult * 2))
		s = self.specific_feature(line)
		#print('line---s', s.shape)
		s = s.unsqueeze(2)
		s = s.unsqueeze(3)
		#print('line---s', s.shape)

		return [f, e0, e1, e2], c, s

class UnetDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(UnetDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		activation = nn.ReLU(True)
		self.ada = AdaIN()

		#self.ha = FN.SA()
		#self.gcf = FN.CA()
		#self.sr = FN.SRM()
		#self.fia = FN.FAM()

		### decoder
		mult = 16
		generator_t = []
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout)]
			self.generator_t = nn.Sequential(*generator_t)

		mult = 16
		generator_d2 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult ), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult)), activation]
		generator_d2 += [nn.Conv2d(ngf * mult , int(ngf * mult/2 ), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 8
		generator_d1 =  [nn.ConvTranspose2d(ngf * mult * 2 , int(ngf * mult ), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult)), activation]
		generator_d1 += [nn.Conv2d(ngf * mult , int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 4
		generator_d0 =  [nn.ConvTranspose2d(ngf * mult * 2 , int(ngf * mult ), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult)), activation]
		generator_d0 += [nn.Conv2d(ngf * mult  , int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d0 = nn.Sequential(*generator_d0)

		final = []
		final += [nn.ReflectionPad2d(2), nn.Conv2d(ngf * mult, ngf, kernel_size=5, padding=0)]
		final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, f, c, s):
		#print('c[4]', c[4].shape, 's', s.shape)
		t = self.ada(c, s)
		t = self.generator_t(t)

		cat = torch.cat((t, f[3]), 1)
		u2 = self.generator_d2(cat)
		#print('u2',u2.shape, 'f[2]',f[2].shape)

		cat = torch.cat((u2, f[2]), 1)
		u1 = self.generator_d1(cat)

		cat = torch.cat((u1, f[1]), 1)
		u0 = self.generator_d0(cat)

		cat = torch.cat((u0, f[0]), 1)
		output = self.final(cat)
		

		return output

##############
## HDR-UNet ##
##############

class HierarchyUnetEncoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HierarchyUnetEncoder, self).__init__()
		self.n_width = n_width
		self.n_height = n_height
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		activation = nn.ReLU(True)
		sub_n_blocks = 1
		### feature extractor
		generator_ef = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		#feture_extract += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, ngf*2, kernel_size=5, padding=0), norm_layer(ngf*2), activation]
		self.generator_ef = nn.Sequential(*generator_ef)
		generator_cf = []
		for i in range(sub_n_blocks):
			generator_cf += [ResnetBlock(ngf , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]

		#self.generator_ef = nn.Sequential(*generator_ef)
		self.generator_cf = nn.Sequential(*generator_cf)

		# encoder
		mult = 1
		
		generator_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		generator_c0 = []
		for i in range(sub_n_blocks):
			generator_c0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]

		self.generator_e0 = nn.Sequential(*generator_e0)
		self.generator_c0 = nn.Sequential(*generator_c0)

		mult = 2
		self.mult = mult
		generator_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		generator_c1 = []
		for i in range(n_blocks):
			generator_c1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]

		self.generator_e1 = nn.Sequential(*generator_e1)
		self.generator_c1 = nn.Sequential(*generator_c1)

		#mult = 8
		#self.mult = mult
		#generator_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		#generator_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_e2 = nn.Sequential(*generator_e2)

		self.generator_sf = nn.Sequential(nn.AvgPool2d(2, stride=2), 
			#nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation,
			nn.AvgPool2d(2, stride=2), 
			nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation
			)
		self.generator_s0 = nn.Sequential(
			#nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult), activation
			)
		self.generator_s1 = nn.Sequential(
			#nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), activation
			)

		### resnet blocks

		#generator_c = []
		#for i in range(n_blocks):
		#	generator_c += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		#	self.generator_c = nn.Sequential(*generator_c)

		#self.specific_feature = nn.Sequential(
		#	nn.Linear((self.ngf * self.mult * 2) * (self.n_height // (self.mult*2)) * (self.n_width // (self.mult*2)), ngf *  self.mult * 2),
		#	nn.LeakyReLU(0.2, True),
		#	nn.Linear(ngf *  self.mult * 2, ngf *  self.mult * 2),
		#	)
		self.specific_feature_sf = SpecificFeature(ngf , ngf , 1)
		self.specific_feature_s0 = SpecificFeature(ngf * mult, ngf * mult, 1)
		self.specific_feature_s1 = SpecificFeature(ngf * mult * 2, ngf * mult * 2, 1)

	def forward(self, input):
		ef = self.generator_ef(input)
		cf = self.generator_cf(ef)
		sf = self.generator_sf(ef)

		e0 = self.generator_e0(cf)
		c0 = self.generator_c0(e0)
		s0 = self.generator_s0(e0)

		e1 = self.generator_e1(c0)
		c1 = self.generator_c1(e1)
		s1 = self.generator_s1(e1)


		#line = s.view(-1, (self.ngf * self.mult * 2) * (self.n_height // (self.mult*2)) * (self.n_width // (self.mult*2)))
		sf = self.specific_feature_sf(sf)
		s0 = self.specific_feature_s0(s0)
		s1 = self.specific_feature_s1(s1)

		#s = s.unsqueeze(2)
		#s = s.unsqueeze(3)

		return [ef, e0, e1], [cf, c0, c1], [sf, s0, s1]


class HierarchyUnetDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HierarchyUnetDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self.ada = AdaIN()
		activation = nn.ReLU(True)
		### decoder
		mult = 4
		generator_t = []
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
			self.generator_t = nn.Sequential(*generator_t)

		#mult = 16
		#generator_d2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 4
		sub_n_blocks = 1
		generator_d1 = [nn.Conv2d(ngf * mult * 2, int(ngf * mult), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult)), activation]
		generator_d1 += [nn.ConvTranspose2d(int(ngf * mult ), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d1 += [nn.Conv2d(int(ngf * mult / 2) , int(ngf * mult / 2) , kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			#print('do rui feng')
			generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 2
		generator_d0 = [nn.Conv2d(ngf * mult* 2, int(ngf * mult), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult )), activation]
		generator_d0 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d0 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d0 = nn.Sequential(*generator_d0)

		conv_skip_1 = [nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), norm_layer(ngf*4), activation]
		conv_skip_0 = [nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), norm_layer(ngf*2), activation]
		conv_skip_f = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		self.conv_skip_1 = nn.Sequential(*conv_skip_1)
		self.conv_skip_0 = nn.Sequential(*conv_skip_0)
		self.conv_skip_f = nn.Sequential(*conv_skip_f)

		final = []
		final += [nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, f, c, s):
		#[e0, e1], [cf, c0, c1], [sf, s0, s1]
		t = self.ada(c[2], s[2])
		#print('s[2]',s[2].shape, 's[1]', s[1].shape)
		dt = self.generator_t(t)
		#skip = self.conv_skip_1(f[2])
		cat = torch.cat((dt, t), 1)
		d1 = self.generator_d1(cat)

		t = self.ada(c[1], s[1])
		#skip = self.conv_skip_0(t)
		cat = torch.cat((d1, t), 1)
		d0 = self.generator_d0(cat)

		t = self.ada(c[0], s[0])
		#skip = self.conv_skip_f(t)
		cat = torch.cat((d0, t), 1)
		output = self.final(cat)

		return output

class HierarchyUnetAddDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HierarchyUnetAddDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self.ada = AdaIN()
		activation = nn.ReLU(True)
		### decoder
		mult = 4
		generator_t = []
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
			self.generator_t = nn.Sequential(*generator_t)

		#mult = 16
		#generator_d2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 4
		sub_n_blocks = 1
		generator_d1 = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult)), activation]
		generator_d1 += [nn.ConvTranspose2d(int(ngf * mult ), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d1 += [nn.Conv2d(int(ngf * mult / 2) , int(ngf * mult / 2) , kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			#print('do rui feng')
			generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 2
		generator_d0 = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult )), activation]
		generator_d0 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d0 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d0 = nn.Sequential(*generator_d0)

		conv_skip_1 = [nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), norm_layer(ngf*4), activation]
		conv_skip_0 = [nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), norm_layer(ngf*2), activation]
		conv_skip_f = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		self.conv_skip_1 = nn.Sequential(*conv_skip_1)
		self.conv_skip_0 = nn.Sequential(*conv_skip_0)
		self.conv_skip_f = nn.Sequential(*conv_skip_f)

		final = []
		final += [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, _, c, se):
		#[e0, e1], [cf, c0, c1], [sf, s0, s1]
		t = self.ada(c[2], se[2])
		#print('s[2]',s[2].shape, 's[1]', s[1].shape)
		dt = self.generator_t(t)
		#skip = self.conv_skip_1(f[2])
		cat = dt + t #+ skip
		d1 = self.generator_d1(cat)

		t = self.ada(c[1], se[1])
		#skip = self.conv_skip_0(f[1])
		cat = d1 + t #+ skip
		d0 = self.generator_d0(cat)

		t = self.ada(c[0], se[0])
		#skip = self.conv_skip_f(f[0])
		cat = d0 + t# + skip
		output = self.final(cat)

		return output


##########################################

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class HierarchySFUnetEncoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HierarchySFUnetEncoder, self).__init__()
		self.n_width = n_width
		self.n_height = n_height
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		activation = nn.ReLU(True)
		sub_n_blocks = 2
		### feature extractor
		generator_ef = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		#generator_cf += [ResnetBlock(ngf , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		#feture_extract += [nn.ReflectionPad2d(2), nn.Conv2d(ngf, ngf*2, kernel_size=5, padding=0), norm_layer(ngf*2), activation]
		#self.generator_ef = nn.Sequential(*generator_ef)
		generator_cf = []
		for i in range(sub_n_blocks):
			generator_cf += [ResnetBlock(ngf , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]

		self.generator_ef = nn.Sequential(*generator_ef)
		self.generator_cf = nn.Sequential(*generator_cf)

		# encoder
		mult = 1
		
		generator_e0 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		generator_e0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]

		generator_c0 = []
		for i in range(sub_n_blocks):
			generator_c0 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]

		self.generator_e0 = nn.Sequential(*generator_e0)
		self.generator_c0 = nn.Sequential(*generator_c0)

		mult = 2
		self.mult = mult
		generator_e1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		generator_e1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		generator_c1 = []
		for i in range(n_blocks):
			generator_c1 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]

		self.generator_e1 = nn.Sequential(*generator_e1)
		self.generator_c1 = nn.Sequential(*generator_c1)

		#mult = 8
		#self.mult = mult
		#generator_e2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
		#generator_e2 += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_e2 = nn.Sequential(*generator_e2)

		self.generator_sf = nn.Sequential(nn.AvgPool2d(2, stride=2), 
			nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation,
			nn.AvgPool2d(2, stride=2), 
			nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation
			)
		self.generator_s0 = nn.Sequential(
			nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult), activation
			)
		self.generator_s1 = nn.Sequential(
			nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), activation,
			nn.AvgPool2d(2, stride=2),
			nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), activation
			)

		### resnet blocks

		#generator_c = []
		#for i in range(n_blocks):
		#	generator_c += [ResnetBlock(ngf * mult * 2, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		#	self.generator_c = nn.Sequential(*generator_c)

		#self.specific_feature = nn.Sequential(
		#	nn.Linear((self.ngf * self.mult * 2) * (self.n_height // (self.mult*2)) * (self.n_width // (self.mult*2)), ngf *  self.mult * 2),
		#	nn.LeakyReLU(0.2, True),
		#	nn.Linear(ngf *  self.mult * 2, ngf *  self.mult * 2),
		#	)
		self.specific_feature_sf = SpecificFeature(ngf , ngf , 1)
		self.specific_feature_s0 = SpecificFeature(ngf * mult, ngf * mult, 1)
		self.specific_feature_s1 = SpecificFeature(ngf * mult * 2, ngf * mult * 2, 1)

	def forward(self, input):
		ef = self.generator_ef(input)
		cf = self.generator_cf(ef)
		sf = self.generator_sf(ef)

		e0 = self.generator_e0(cf)
		c0 = self.generator_c0(e0)
		s0 = self.generator_s0(e0)

		e1 = self.generator_e1(c0)
		c1 = self.generator_c1(e1)
		s1 = self.generator_s1(e1)


		#line = s.view(-1, (self.ngf * self.mult * 2) * (self.n_height // (self.mult*2)) * (self.n_width // (self.mult*2)))
		sf = self.specific_feature_sf(sf)
		s0 = self.specific_feature_s0(s0)
		s1 = self.specific_feature_s1(s1)

		#s = s.unsqueeze(2)
		#s = s.unsqueeze(3)

		return [ef, e0, e1], [cf, c0, c1], [sf, s0, s1]

class HierarchySFUnetAddDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, n_blocks=3, norm_layer=nn.InstanceNorm2d, use_dropout=False, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HierarchySFUnetAddDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.fsm_f = FSM(self.ngf    , 4, bias=False, norm_layer=norm_layer)
		self.fsm_0 = FSM(self.ngf * 2, 4, bias=False, norm_layer=norm_layer)
		self.fsm_1 = FSM(self.ngf * 4, 4, bias=False, norm_layer=norm_layer)

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self.ada = AdaIN()
		activation = nn.ReLU(True)
		### decoder
		mult = 4
		generator_t = []
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
			self.generator_t = nn.Sequential(*generator_t)

		#mult = 16
		#generator_d2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		#self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 4
		sub_n_blocks = 2
		generator_d1 = []
		generator_d1 += [ResnetBlock(ngf * mult , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		generator_d1 += [nn.ConvTranspose2d(int(ngf * mult ), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d1 += [nn.Conv2d(int(ngf * mult / 2) , int(ngf * mult / 2) , kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			#print('do rui feng')
			generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 2
		generator_d0 = []
		generator_d0 += [ResnetBlock(ngf * mult , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		generator_d0 += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		#generator_d0 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), norm_layer(int(ngf * mult / 2)), activation]
		for i in range(sub_n_blocks):
			generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.generator_d0 = nn.Sequential(*generator_d0)

		conv_skip_1 = [ResnetBlock(ngf * 4 , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		conv_skip_0 = [ResnetBlock(ngf * 2 , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		conv_skip_f = [ResnetBlock(ngf  , padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout), activation]
		self.conv_skip_1 = nn.Sequential(*conv_skip_1)
		self.conv_skip_0 = nn.Sequential(*conv_skip_0)
		self.conv_skip_f = nn.Sequential(*conv_skip_f)

		final = []
		#final += [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]
		final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, f, c, s):
		#[e0, e1], [cf, c0, c1], [sf, s0, s1]
		
		t = self.ada(c[2], s[2])
		#print('s[2]',s[2].shape, 's[1]', s[1].shape)
		dt = self.generator_t(c[2])
		skip = f[2]#self.conv_skip_1(f[2])
		fuse, score_att_1 = self.fsm_1(skip, t)
		d1 = self.generator_d1(fuse + dt)

		
		t = self.ada(c[1], s[1])
		skip = f[1]#self.conv_skip_0(f[1])
		fuse, score_att_0 = self.fsm_0(skip, t)
		d0 = self.generator_d0(fuse + d1)


		t = self.ada(c[0], s[0])
		skip = f[0]#self.conv_skip_f(f[0])
		fuse, score_att_f = self.fsm_f(skip, t)
		output = self.final(fuse + d0)

		return output

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/


class ResnetNoadaDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(ResnetNoadaDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		#self.ada = AdaIN()
		activation = nn.ReLU(True)
		### decoder
		mult = 16
		generator_t = []
		for i in range(n_blocks):
			generator_t += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=activation, use_dropout=use_dropout)]
			self.generator_t = nn.Sequential(*generator_t)
		mult = 16
		generator_d2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 8
		generator_d1 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 4
		generator_d0 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d0 = nn.Sequential(*generator_d0)

		#final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf*2, ngf, kernel_size=5, padding=0)]
		final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf*2, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, c, s):
		t = c + s
		dt = self.generator_t(t)
		d2 = self.generator_d2(dt)
		d1 = self.generator_d1(d2)
		d0 = self.generator_d0(d1)
		output = self.final(d0)

		return output


class UnetNoadaDecoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, n_width=256, n_height=128, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(UnetNoadaDecoder, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		activation = nn.ReLU(True)
		#self.ada = AdaIN()

			### decoder
		mult = 16
		generator_d2 = [nn.ConvTranspose2d(ngf * mult * 2, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		generator_d2 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d2 = nn.Sequential(*generator_d2)

		mult = 8
		generator_d1 = [nn.ConvTranspose2d(ngf * mult * 2 , int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		generator_d1 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d1 = nn.Sequential(*generator_d1)

		mult = 4
		generator_d0 = [nn.ConvTranspose2d(ngf * mult * 2 , int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, output_padding=0), norm_layer(int(ngf * mult / 2)), activation]
		generator_d0 += [ResnetBlock(int(ngf * mult / 2), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
		self.generator_d0 = nn.Sequential(*generator_d0)

		final = [nn.ReflectionPad2d(2), nn.Conv2d(ngf * mult, ngf, kernel_size=5, padding=0)]
		final += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0), nn.Tanh()]
		self.final = nn.Sequential(*final)

	def forward(self, c, s):
		#print('c[4]', c[4].shape, 's', s.shape)
		#t = self.ada(c[4], s)
		t = c[4]+s
		cat = torch.cat((t, c[3]), 1)
		u2 = self.generator_d2(cat)

		cat = torch.cat((u2, c[2]), 1)
		u1 = self.generator_d1(cat)

		cat = torch.cat((u1, c[1]), 1)
		u0 = self.generator_d0(cat)

		cat = torch.cat((u0, c[0]), 1)
		output = self.final(cat)
		

		return output

#===============================
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        r = 1
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, int(dim/r), kernel_size=3, padding=p),
                       norm_layer(int(dim/r)),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(int(dim/r), dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out





# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, use_feature=False):
		super(NLayerDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		self.use_feature = use_feature
		kw = 4
		padw = 1
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]
		
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]
		self.model_f = nn.Sequential(*sequence)

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 16)
		sequence2 = [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
			          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		
		sequence2 += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
		
		if use_sigmoid:
			sequence2 += [nn.Sigmoid()]
		
		self.model = nn.Sequential(*sequence2)
	
	def forward(self, input):
		#print(self.use_feature)
		if not self.use_feature:
			return self.model(self.model_f(input))
		else:
			#print('using feature map lossing')
			return self.model_f(input)



class PixelDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
		super(PixelDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		self.net = [
			nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
			norm_layer(ndf * 2),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
		
		if use_sigmoid:
			self.net.append(nn.Sigmoid())
		
		self.net = nn.Sequential(*self.net)
	
	def forward(self, input):
		return self.net(input)


class Classifier(nn.Module):
	def __init__(self, input_nc, ndf, norm_layer=nn.InstanceNorm2d):
		super(Classifier, self).__init__()
		
		kw = 3
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2),
			nn.LeakyReLU(0.2, True)
		]
		
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(3):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				          kernel_size=kw, stride=2),
				norm_layer(ndf * nf_mult, affine=True),
				nn.LeakyReLU(0.2, True)
			]
		self.before_linear = nn.Sequential(*sequence)
		
		sequence = [
			nn.Linear(ndf * nf_mult, 1024),
			nn.Linear(1024, 10)
		]
		
		self.after_linear = nn.Sequential(*sequence)
	
	def forward(self, x):
		bs = x.size(0)
		out = self.after_linear(self.before_linear(x).view(bs, -1))
		return out
#       return nn.functional.log_softmax(out, dim=1)
