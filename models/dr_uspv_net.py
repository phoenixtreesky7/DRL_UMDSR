import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from epdn import epdn_networks
from invo.involution import Involution2d
import torch.nn.functional as F
from models.actnorm import ActNorm2d
from models.sparse_switchable_norm import SSN2d
from models.switchable_norm import SwitchNorm2d
from models.layer_norm import LayerNormalization_NoGB as LayerNorm
from models.adain import adaptive_instance_normalization as adain
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'actnorm':
		norm_layer = functools.partial(ActNorm2d)
		print('normalization: actnorm')
	elif norm_type == 'sswitch':
		norm_layer = functools.partial(SSN2d)
		print('normalization: sparse-switchnorm')
	elif norm_type == 'switch':
		norm_layer = functools.partial(SwitchNorm2d)
		print('normalization: swithchnorm')
	elif norm_type == 'layer':
		norm_layer = functools.partial(LayerNorm)
		print('normalization: layernorm')
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
		print('lr_policy = step')
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
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

###########################################
###########################################
## Selections of the Encoder and Decoder ##
###########################################
###########################################
def define_G_E_A(input_nc, output_nc, ngf, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'dr_at':
		netG = DRUNetPCSAMEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'dr_ls':
		netG = DRLSEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'dr_v2':
		netG = DRUNetV2EncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_ca':
		netG = DRUNetCAMEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr':
		netG = DREncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_unet':
		print('current networks: dr_unet')
		netG = DRUNetEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

def define_G_D_A(input_nc, output_nc, ngf, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'dr_at':
		netG = DRUNetPCSAMDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'dr_ls':
		netG = DRLSDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'dr_v2':
		netG = DRUNetV2DecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_ca':
		netG = DRUNetCAMDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr':
		netG = DRDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_unet':
		netG = DRUNetDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

########################################

def define_G_E_B(input_nc, output_nc, ngf, n_blocks, haze_layer, which_model_netG, ca_type, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'dr_at':
		netG = DRUNetPCSAMEncoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'dr_ls':
		netG = DRLSEncoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'dr_v2':
		netG = DRUNetV2EncoderHazy(input_nc, output_nc, ngf, haze_layer, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_ca':
		netG = DRUNetCAMEncoderHazy(input_nc, output_nc, ngf, haze_layer, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr':
		netG = DREncoderHazy(input_nc, output_nc, ngf, haze_layer, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_unet':
		netG = DRUNetEncoderHazy(input_nc, output_nc, ngf, haze_layer, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, ca_type=ca_type, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

def define_G_D_B(input_nc, output_nc, ngf, n_blocks, haze_layer, which_model_netG, fuse_model, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'dr_at':
		netG = DRUNetPCSAMDecoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'dr_ls':
		netG = DRLSDecoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'dr_v2':
		netG = DRUNetV2DecoderHazy(input_nc, output_nc, ngf, haze_layer, fuse_model=fuse_model, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_ca':
		netG = DRUNetCAMDecoderHazy(input_nc, output_nc, ngf, haze_layer, fuse_model=fuse_model, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr':
		netG = DRDecoderHazy(input_nc, output_nc, ngf, haze_layer, fuse_model=fuse_model, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'dr_unet':
		netG = DRUNetDecoderHazy(input_nc, output_nc, ngf, haze_layer, fuse_model=fuse_model, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

#######################################

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
	netD = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netD == 'basic':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'nl':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'cd':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'fd':
		netD = NLayerFeatureDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'pixel':
		netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' %
		                          which_model_netD)
	return init_net(netD, init_type, gpu_ids)


###########################################
###########################################
##              Sub Functions            ##
###########################################
###########################################

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


class SFT_layer(nn.Module):
    def __init__(self, c_nc, h_nc):
        super(SFT_layer, self).__init__()
        Relu = nn.LeakyReLU(0.2, True)

        condition_conv1 = nn.Conv2d(h_nc, c_nc, kernel_size=3, stride=1, padding=1)
        condition_conv2 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        condition_conv3 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)

        conditon_conv = [condition_conv1, Relu, condition_conv2, Relu, condition_conv3, Relu]
        self.condition_conv = nn.Sequential(*conditon_conv)

        scale_conv1 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        scale_conv2 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        scale_conv = [scale_conv1, Relu, scale_conv2, Relu]
        self.scale_conv = nn.Sequential(*scale_conv)

        sift_conv1 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        sift_conv2 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        sift_conv = [sift_conv1, Relu, sift_conv2, Relu]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, c, h):
        h_condition = self.condition_conv(h)
        scaled_feature = self.scale_conv(h_condition) * c
        sifted_feature = scaled_feature + self.sift_conv(h_condition)

        return sifted_feature

class Dehaze(nn.Module):
    def __init__(self):
        super(Dehaze, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
        self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1050 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        #refine3 = [nn.ReflectionPad2d(3)]
        refine3 = [nn.Conv2d(20+5, 3, kernel_size=3, stride=1, padding=1)]
        refine3 += [nn.Tanh()]
        self.refine3 = nn.Sequential(*refine3)
        self.upsample = F.upsample_bilinear

        self.batch20 = ActNorm2d(20)
        self.batch1 = ActNorm2d(1)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 32)

        x102 = F.avg_pool2d(dehaze, 16)

        x103 = F.avg_pool2d(dehaze, 8)

        x104 = F.avg_pool2d(dehaze, 4)

        x105 = F.avg_pool2d(dehaze, 2)

        x1010 = self.upsample(self.relu((self.conv1010(x101))),size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))),size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))),size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))),size=shape_out)
        x1050 = self.upsample(self.relu((self.conv1050(x105))),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x1050, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze


class CAM(nn.Module):
    def __init__(self, in_nc, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_nc, int(in_nc/ratio), 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(int(in_nc/ratio), in_nc, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = (avg_out + max_out)
        return self.sigmoid(out)*x + x

class CAM_no_residual(nn.Module):
    def __init__(self, in_nc, ratio=16):
        super(CAM_no_residual, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_nc, int(in_nc/ratio), 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(int(in_nc/ratio), in_nc, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = (avg_out + max_out)
        return self.sigmoid(out)*x #+ x

class SAM_no_residual(nn.Module):
    def __init__(self, kernel_size=5):
        super(SAM_no_residual, self).__init__()
        assert kernel_size in (3, 5), 'kernel size must be 3 or 5'
        padding = 2 if kernel_size == 5 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        avg_out = torch.mean(input, dim=1, keepdim=True)
        max_out, _ = torch.max(input, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        out = self.sigmoid(x)*input #+ input
        return out

class PCSAM(nn.Module):
	def __init__(self, in_nc, ratio=16, kernel_size=5):
		super(PCSAM, self).__init__()
		self.cam = CAM_no_residual(in_nc, ratio)
		self.sam = SAM_no_residual(kernel_size)
		#self.conv1 = nn.Conv2d(in_nc*2, in_nc, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x_cam = self.cam(x)
		x_sam = self.sam(x)
		out = self.relu(x_cam + x_sam + x) # self.conv1(torch.cat((x_cam, x_sam), 1))
		return out

class CBAM(nn.Module):
	def __init__(self, in_nc, ratio=16, kernel_size=5):
		super(CBAM, self).__init__()
		self.cam = CAM_no_residual(in_nc, ratio)
		self.sam = SAM_no_residual(kernel_size)
		#self.conv1 = nn.Conv2d(in_nc, in_nc, kernel_size=1, padding=0, bias=False)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x_cam = self.cam(x)
		x_sam = self.sam(x_cam)
		out = self.relu(x_sam + x)
		return out

########################################################################

########################      #######################################
##############################################      #################
#####################################################################
##                            DR + Long-skip                       ##
##      Disentangled Reperesentation with long-skip connection     ##
#####################################################################
##############################################      #################
########################      #######################################

## Encoder for Clear Image ##

class DRLSEncoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRLSEncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		#encode_contfe += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(ngf * mult *2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		ec5 = self.encode_cont5(ec4)
		#print('ec5', ec5.shape)
		return [ecfe, ec1, [], [], [], ec5]




##########################################################
## Encoder for Clear image 
##########################################################
class DRLSDecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRLSDecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		n_downsampling = 3
		mult = 32
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)

		mult = 2 * 16
		decoder_up5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up5 += [nn.Conv2d(int(ngf * mult /2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up5 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up5 = nn.Sequential(*decoder_up5)

		mult = 2 * 8
		decoder_up4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up4 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up4 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up4 = nn.Sequential(*decoder_up4)

		mult = 2 * 4
		decoder_up3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up3 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up3 = nn.Sequential(*decoder_up3)

		mult = 2 * 2
		decoder_up2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up2 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up2 = nn.Sequential(*decoder_up2)

		mult = 2 * 2
		decoder_up1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up1 = nn.Sequential(*decoder_up1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]  #+ int(ngf/2)
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]  #+ int(ngf/2)
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#
		#skip_fn = [nn.Conv2d(3, int(ngf), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)


		#self.dehaze = Dehaze()

	
	def forward(self, input, c):
		dft = self.decode_ft(c[5])

		#concat = torch.cat((c[5], dft), 1) # self.skip_3(c[3])
		dup5 = self.decoder_up5(dft)

		#concat = torch.cat((c[4], dup5), 1)  # self.skip_2(c[2])
		dup4 = self.decoder_up4(dup5)

		#concat = torch.cat((c[3], dup4), 1) # self.skip_3(c[3])
		dup3 = self.decoder_up3(dup4)

		#concat = torch.cat((c[2], dup3), 1)  # self.skip_2(c[2])
		dup2 = self.decoder_up2(dup3)

		concat = torch.cat((c[1], dup2), 1)  # self.skip_1(c[1])
		dup1 = self.decoder_up1(concat)

		concat = torch.cat((c[0], dup1), 1) # self.skip_fe(c[0]), self.skip_fn(input)
		#print(concat.shape)
		out = self.decoder_fn(concat)

		return out#, dehaze


########################################################################


class DRLSEncoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer=2, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, ca_type='cross_ca', padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRLSEncoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.hl = haze_layer
		self.ca_type = ca_type
		self.n_blocks = n_blocks
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		#self.sa = SpatialAttention(7)

		## Content Encoder ##
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		##################
		## Haze Encoder ##
		##################
		encode_hazefe = [nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_hazefe = nn.Sequential(*encode_hazefe)

		# hazy style layer 1 #
		mult = 1
		encode_haze1 = [nn.AvgPool2d(2, stride=2)]
		
		encode_haze1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		self.encode_haze1 = nn.Sequential(*encode_haze1)


		# hazy style layer 2 #
		mult = 2
		encode_haze2 = [nn.AvgPool2d(2, stride=2)]
		ratio = 8
		encode_haze2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		self.encode_haze2 = nn.Sequential(*encode_haze2)
		#self.channel_att_2 = ChannelAttention(int(ngf * mult * 2), int(ngf * mult * 2/ratio))

		## 
		mult = 4
		
		haze_feature_l1 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		mult = 8
		
		haze_feature_l2 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		mult = 16
		
		haze_feature_l3 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		self.haze_feature_l1 = nn.Sequential(*haze_feature_l1)
		self.haze_feature_l2 = nn.Sequential(*haze_feature_l2)
		self.haze_feature_l3 = nn.Sequential(*haze_feature_l3)

		



	def forward(self, input):
		ecfe = self.encode_contfe(input)
		#print('encoder-hazy: input', input.shape)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		#print('encoder-hazy: ec4', ec4.shape)
		ec5 = self.encode_cont5(ec4)
		#print('encoder-hazy: ec5', ec5.shape)

		ehfe = self.encode_hazefe(input)

		eh1 = self.encode_haze1(ehfe)
		eh2 = self.encode_haze2(eh1)
		eh3 = self.haze_feature_l1(eh2)
		eh4 = self.haze_feature_l2(eh3)
		eh5 = self.haze_feature_l3(eh4)

		#print('encoder-hazy: eh5+conv.', eh5.shape)


		


		return [ecfe, ec1, [], [], [], ec5], [[], [], [], [], [], eh5]



##########################################################
class DRLSDecoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer = 2, fuse_model='csfm', norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRLSDecoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.fuse_model = fuse_model
		self.hl = haze_layer
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		n_downsampling = 3
		
		mult = 32
		sft_5 = [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias)]
		self.sft_5 = nn.Sequential(*sft_5)

		
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)


		mult = 2 * 16
		#decoder_hf5 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		decoder_hf5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf5 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf5 = nn.Sequential(*decoder_hf5)

		#decoder_hf4 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]

		mult = 2 * 8
		decoder_hf4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf4 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf4 = nn.Sequential(*decoder_hf4)

		#decoder_hf3 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		mult = 2 * 4
		decoder_hf3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf3 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf3 = nn.Sequential(*decoder_hf3)

		mult = 2 * 2
		decoder_hf2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf2 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf2 = nn.Sequential(*decoder_hf2)

		mult = 2 * 2
		decoder_hf1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf1 = nn.Sequential(*decoder_hf1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#skip_fn = [nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)
		#skip_fe = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fe = nn.Sequential(*skip_fe)
		#skip_1 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 2), nn.ReLU(True)]
		#self.skip_1 = nn.Sequential(*skip_1)
		#skip_2 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 4), nn.ReLU(True)]
		#self.skip_2 = nn.Sequential(*skip_2)
		#skip_3 = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 8), nn.ReLU(True)]
		#self.skip_3 = nn.Sequential(*skip_3)
	
	def forward(self, input, c, h):
		
		
		fuse_5 = c[5]
		fuse_5 = adain(fuse_5, h[5])
		#fuse_5 = self.sft_5(torch.cat((c[5], h[5]), 1))


		dft = self.decode_ft(fuse_5)
		#print('fuse_5', fuse_5.shape, 'dft', dft.shape)
		#concat = torch.cat((fuse_5, dft), 1)
		dhf5 = self.decoder_hf5(dft)

		#print('fuse_4', fuse_4.shape, 'dhf5', dhf5.shape)
		#concat = torch.cat((c[4], dhf5), 1)
		dhf4 = self.decoder_hf4(dhf5)

		#print('fuse_3', fuse_4.shape, 'dhf4', dhf5.shape)
		#concat = torch.cat((c[3], dhf4), 1)
		dhf3 = self.decoder_hf3(dhf4)

		#concat = torch.cat((c[2], dhf3), 1)
		dhf2 = self.decoder_hf2(dhf3)

		concat = torch.cat((c[1], dhf2), 1)
		dhf1 = self.decoder_hf1(concat)
		#print(c[0].shape, self.skip_fn(input).shape, dhf1.shape, dhf2.shape, dhf3.shape, dhf4.shape, dhf5.shape)

		concat = torch.cat((c[0], dhf1), 1)
		out = self.decoder_fn(concat) 
		return out


#--------------------------------------------------------------------

########################      #######################################
##############################################      #################
#####################################################################
##                             DR + UNet                           ##
##       Disentangled Reperesentation with U-Net Architecture      ##
#####################################################################
##############################################      #################
########################      #######################################
## Encoder for Clear Image ##

class DRUNetV2EncoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetV2EncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		norm_layer = nn.InstanceNorm2d
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		#encode_contfe += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(ngf * mult *2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont1 += [CAM(ngf*mult*2, 16)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont2 += [CAM(ngf*mult*2, 16)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont3 += [CAM(ngf*mult*2, 16)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont4 += [CAM(ngf*mult*2, 16)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont5 += [CAM(ngf*mult*2, 16)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		mult = 32
		encode_et = []
		for i in range(2):
			encode_et += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decode_ft += [CAM(ngf*mult, 16)]

		self.encode_et = nn.Sequential(*encode_et)

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		ec5 = self.encode_cont5(ec4)
		#print('ec5', ec5.shape)
		return [ecfe, ec1, ec2, ec3, ec4, self.encode_et(ec5)]




##########################################################
## Encoder for Clear image 
##########################################################
class DRUNetV2DecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetV2DecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		
		n_downsampling = 3
		mult = 32
		decode_ft = []
		for i in range(2):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decode_ft += [CAM(ngf*mult, 16)]

		self.decode_ft = nn.Sequential(*decode_ft)

		mult = 2 * 32
		decoder_up5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up5 += [nn.Conv2d(int(ngf * mult /2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up5 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_up5 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up5 = nn.Sequential(*decoder_up5)

		mult = 2 * 16
		decoder_up4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up4 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up4 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_up4 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up4 = nn.Sequential(*decoder_up4)

		mult = 2 * 8
		decoder_up3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up3 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_up3 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up3 = nn.Sequential(*decoder_up3)

		mult = 2 * 4
		decoder_up2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up2 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_up2 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up2 = nn.Sequential(*decoder_up2)

		mult = 2 * 2
		decoder_up1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
		decoder_up1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_up1 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up1 = nn.Sequential(*decoder_up1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]  #+ int(ngf/2)
		decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]  #+ int(ngf/2)
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult/2), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#
		#skip_fn = [nn.Conv2d(3, int(ngf), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)


		#self.dehaze = Dehaze()

	
	def forward(self, input, c):
		dft = self.decode_ft(c[5])

		concat = torch.cat((c[5], dft), 1) # self.skip_3(c[3])
		dup5 = self.decoder_up5(concat)

		concat = torch.cat((c[4], dup5), 1)  # self.skip_2(c[2])
		dup4 = self.decoder_up4(concat)

		concat = torch.cat((c[3], dup4), 1) # self.skip_3(c[3])
		dup3 = self.decoder_up3(concat)

		concat = torch.cat((c[2], dup3), 1)  # self.skip_2(c[2])
		dup2 = self.decoder_up2(concat)

		concat = torch.cat((c[1], dup2), 1)  # self.skip_1(c[1])
		dup1 = self.decoder_up1(concat)

		concat = torch.cat((c[0], dup1), 1) # self.skip_fe(c[0]), self.skip_fn(input)
		#print(concat.shape)
		out = self.decoder_fn(concat)

		return out#, dehaze


########################################################################


class DRUNetV2EncoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer=2, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, ca_type='cross_ca', padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetV2EncoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.hl = haze_layer
		self.ca_type = ca_type
		self.n_blocks = n_blocks
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		norm_layer = nn.InstanceNorm2d

		## Content Encoder ##
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont1 += [PCSAM(ngf * mult*2, 16, 3)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont2 += [CAM(ngf * mult*2, 16)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont3 += [CAM(ngf * mult*2, 16)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont4 += [CAM(ngf * mult*2, 16)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont5 += [CAM(ngf * mult*2, 16)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		mult = 32
		encode_ct = []
		for i in range(n_blocks):
			encode_ct += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		#encode_ct += [CAM(ngf * mult, 16)]
		self.encode_ct = nn.Sequential(*encode_ct)

		##################
		## Haze Encoder ##
		##################
		encode_hazefe = [nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]

		self.encode_hazefe = nn.Sequential(*encode_hazefe)

		# hazy style layer 1 #
		mult = 1
		encode_haze1 = [nn.AvgPool2d(2, stride=2)]
		
		encode_haze1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		#encode_haze1 += [PCSAM(ngf * mult*2, 16, 3)]
		self.encode_haze1 = nn.Sequential(*encode_haze1)


		# hazy style layer 2 #
		mult = 2
		encode_haze2 = [nn.AvgPool2d(2, stride=2)]
		ratio = 8
		encode_haze2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		#encode_haze2 += [CAM(ngf * mult*2, 16)]
		self.encode_haze2 = nn.Sequential(*encode_haze2)
		#self.channel_att_2 = ChannelAttention(int(ngf * mult * 2), int(ngf * mult * 2/ratio))

		## 
		mult = 4
		
		haze_feature_l1 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]
		#haze_feature_l1 += [CAM(ngf * mult*2, 16)]

		mult = 8
		
		haze_feature_l2 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]
		#haze_feature_l2 += [CAM(ngf * mult*2, 16)]

		mult = 16
		
		haze_feature_l3 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]
		#haze_feature_l3 += [CAM(ngf * mult*2, 16)]

		self.haze_feature_l1 = nn.Sequential(*haze_feature_l1)
		self.haze_feature_l2 = nn.Sequential(*haze_feature_l2)
		self.haze_feature_l3 = nn.Sequential(*haze_feature_l3)


		mult = 32
		encode_ht = []
		for i in range(2):
			encode_ht += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		#encode_ht += [CAM(ngf * mult, 16)]
		self.encode_ht = nn.Sequential(*encode_ht)


	def forward(self, input):
		ecfe = self.encode_contfe(input)
		#print('encoder-hazy: input', input.shape)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		#print('encoder-hazy: ec4', ec4.shape)
		ec5 = self.encode_cont5(ec4)
		#print('encoder-hazy: ec5', ec5.shape)

		ehfe = self.encode_hazefe(input)

		eh1 = self.encode_haze1(ehfe)
		eh2 = self.encode_haze2(eh1)
		eh3 = self.haze_feature_l1(eh2)
		eh4 = self.haze_feature_l2(eh3)
		eh5 = self.haze_feature_l3(eh4)

		#print('encoder-hazy: eh5+conv.', eh5.shape)


		return [ecfe, ec1, ec2, ec3, ec4, self.encode_ct(ec5)], [[], [], [], [], [], self.encode_ht(eh5)]



##########################################################
class DRUNetV2DecoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer = 2, fuse_model='csfm', norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetV2DecoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.fuse_model = fuse_model
		self.hl = haze_layer
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		n_downsampling = 3
		
		mult = 32
		#sft_5 = [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias)]
		#self.sft_5 = nn.Sequential(*sft_5)

		
		decode_ft = []
		for i in range(2):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		#decode_ft += [CAM(ngf * mult, 16)]
		self.decode_ft = nn.Sequential(*decode_ft)


		mult = 2 * 32
		#decoder_hf5 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		decoder_hf5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf5 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_hf5 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf5 = nn.Sequential(*decoder_hf5)

		#decoder_hf4 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]

		mult = 2 * 16
		decoder_hf4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf4 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_hf4 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf4 = nn.Sequential(*decoder_hf4)

		#decoder_hf3 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		mult = 2 * 8
		decoder_hf3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf3 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_hf3 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf3 = nn.Sequential(*decoder_hf3)

		mult = 2 * 4
		decoder_hf2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf2 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_hf2 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf2 = nn.Sequential(*decoder_hf2)

		mult = 2 * 2
		decoder_hf1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
		decoder_hf1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_hf1 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf1 = nn.Sequential(*decoder_hf1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]
		decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult/2), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#skip_fn = [nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)
		#skip_fe = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fe = nn.Sequential(*skip_fe)
		#skip_1 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 2), nn.ReLU(True)]
		#self.skip_1 = nn.Sequential(*skip_1)
		#skip_2 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 4), nn.ReLU(True)]
		#self.skip_2 = nn.Sequential(*skip_2)
		#skip_3 = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 8), nn.ReLU(True)]
		#self.skip_3 = nn.Sequential(*skip_3)
	
	def forward(self, input, c, h):
		
		
		fuse_5 = c[5]
		fuse_5 = adain(fuse_5, h[5])
		#fuse_5 = self.sft_5(torch.cat((c[5], h[5]), 1))


		dft = self.decode_ft(c[5])
		#print('fuse_5', fuse_5.shape, 'dft', dft.shape)
		concat = torch.cat((fuse_5, dft), 1)
		dhf5 = self.decoder_hf5(concat)

		#print('fuse_4', fuse_4.shape, 'dhf5', dhf5.shape)
		concat = torch.cat((c[4], dhf5), 1)
		dhf4 = self.decoder_hf4(concat)

		#print('fuse_3', fuse_4.shape, 'dhf4', dhf5.shape)
		concat = torch.cat((c[3], dhf4), 1)
		dhf3 = self.decoder_hf3(concat)

		concat = torch.cat((c[2], dhf3), 1)
		dhf2 = self.decoder_hf2(concat)

		concat = torch.cat((c[1], dhf2), 1)
		dhf1 = self.decoder_hf1(concat)
		#print(c[0].shape, self.skip_fn(input).shape, dhf1.shape, dhf2.shape, dhf3.shape, dhf4.shape, dhf5.shape)

		concat = torch.cat((c[0], dhf1), 1)
		out = self.decoder_fn(concat) 
		return out








#--------------------------------------------------------------------

########################      #######################################
##############################################      #################
#####################################################################
##                             DR + UNet                           ##
##       Disentangled Reperesentation with U-Net Architecture      ##
#####################################################################
##############################################      #################
########################      #######################################
## Encoder for Clear Image ##

class DRUNetCAMEncoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetCAMEncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		norm_layer = nn.InstanceNorm2d
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		#encode_contfe += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(ngf * mult *2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont1 += [PCSAM(ngf*mult*2, 16, 3)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont2 += [CAM(ngf*mult*2, 16)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont3 += [CAM(ngf*mult*2, 16)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont4 += [CAM(ngf*mult*2, 16)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont5 += [CAM(ngf*mult*2, 16)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		mult = 32
		encode_et = []
		for i in range(2):
			encode_et += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decode_ft += [CAM(ngf*mult, 16)]

		self.encode_et = nn.Sequential(*encode_et)

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		ec5 = self.encode_cont5(ec4)
		#print('ec5', ec5.shape)
		return [ecfe, ec1, ec2, ec3, ec4, self.encode_et(ec5)]




##########################################################
## Encoder for Clear image 
##########################################################
class DRUNetCAMDecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetCAMDecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		
		n_downsampling = 3
		mult = 32
		decode_ft = []
		for i in range(2):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decode_ft += [CAM(ngf*mult, 16)]

		self.decode_ft = nn.Sequential(*decode_ft)

		mult = 2 * 32
		decoder_up5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up5 += [nn.Conv2d(int(ngf * mult /2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up5 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_up5 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up5 = nn.Sequential(*decoder_up5)

		mult = 2 * 16
		decoder_up4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up4 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up4 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_up4 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up4 = nn.Sequential(*decoder_up4)

		mult = 2 * 8
		decoder_up3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up3 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_up3 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up3 = nn.Sequential(*decoder_up3)

		mult = 2 * 4
		decoder_up2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up2 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_up2 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up2 = nn.Sequential(*decoder_up2)

		mult = 2 * 2
		decoder_up1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
		decoder_up1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_up1 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_up1 = nn.Sequential(*decoder_up1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]  #+ int(ngf/2)
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]  #+ int(ngf/2)
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#
		#skip_fn = [nn.Conv2d(3, int(ngf), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)


		#self.dehaze = Dehaze()

	
	def forward(self, input, c):
		dft = self.decode_ft(c[5])

		concat = torch.cat((c[5], dft), 1) # self.skip_3(c[3])
		dup5 = self.decoder_up5(concat)

		concat = torch.cat((c[4], dup5), 1)  # self.skip_2(c[2])
		dup4 = self.decoder_up4(concat)

		concat = torch.cat((c[3], dup4), 1) # self.skip_3(c[3])
		dup3 = self.decoder_up3(concat)

		concat = torch.cat((c[2], dup3), 1)  # self.skip_2(c[2])
		dup2 = self.decoder_up2(concat)

		concat = torch.cat((c[1], dup2), 1)  # self.skip_1(c[1])
		dup1 = self.decoder_up1(concat)

		concat = torch.cat((c[0], dup1), 1) # self.skip_fe(c[0]), self.skip_fn(input)
		#print(concat.shape)
		out = self.decoder_fn(concat)

		return out#, dehaze


########################################################################


class DRUNetCAMEncoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer=2, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, ca_type='cross_ca', padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetCAMEncoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.hl = haze_layer
		self.ca_type = ca_type
		self.n_blocks = n_blocks
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		norm_layer = nn.InstanceNorm2d

		## Content Encoder ##
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont1 += [PCSAM(ngf * mult*2, 16, 3)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont2 += [CAM(ngf * mult*2, 16)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont3 += [CAM(ngf * mult*2, 16)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont4 += [CAM(ngf * mult*2, 16)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		encode_cont5 += [CAM(ngf * mult*2, 16)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		mult = 32
		encode_ct = []
		for i in range(n_blocks):
			encode_ct += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		#encode_ct += [CAM(ngf * mult, 16)]
		self.encode_ct = nn.Sequential(*encode_ct)

		##################
		## Haze Encoder ##
		##################
		encode_hazefe = [nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]

		self.encode_hazefe = nn.Sequential(*encode_hazefe)

		# hazy style layer 1 #
		mult = 1
		encode_haze1 = [nn.AvgPool2d(2, stride=2)]
		
		encode_haze1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		#encode_haze1 += [PCSAM(ngf * mult*2, 16, 3)]
		self.encode_haze1 = nn.Sequential(*encode_haze1)


		# hazy style layer 2 #
		mult = 2
		encode_haze2 = [nn.AvgPool2d(2, stride=2)]
		ratio = 8
		encode_haze2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		encode_haze2 += [CAM(ngf * mult*2, 16)]
		self.encode_haze2 = nn.Sequential(*encode_haze2)
		#self.channel_att_2 = ChannelAttention(int(ngf * mult * 2), int(ngf * mult * 2/ratio))

		## 
		mult = 4
		
		haze_feature_l1 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]
		haze_feature_l1 += [CAM(ngf * mult*2, 16)]

		mult = 8
		
		haze_feature_l2 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]
		haze_feature_l2 += [CAM(ngf * mult*2, 16)]

		mult = 16
		
		haze_feature_l3 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]
		haze_feature_l3 += [CAM(ngf * mult*2, 16)]

		self.haze_feature_l1 = nn.Sequential(*haze_feature_l1)
		self.haze_feature_l2 = nn.Sequential(*haze_feature_l2)
		self.haze_feature_l3 = nn.Sequential(*haze_feature_l3)


		mult = 32
		encode_ht = []
		for i in range(2):
			encode_ht += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		#encode_ht += [CAM(ngf * mult, 16)]
		self.encode_ht = nn.Sequential(*encode_ht)


	def forward(self, input):
		ecfe = self.encode_contfe(input)
		#print('encoder-hazy: input', input.shape)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		#print('encoder-hazy: ec4', ec4.shape)
		ec5 = self.encode_cont5(ec4)
		#print('encoder-hazy: ec5', ec5.shape)

		ehfe = self.encode_hazefe(input)

		eh1 = self.encode_haze1(ehfe)
		eh2 = self.encode_haze2(eh1)
		eh3 = self.haze_feature_l1(eh2)
		eh4 = self.haze_feature_l2(eh3)
		eh5 = self.haze_feature_l3(eh4)

		#print('encoder-hazy: eh5+conv.', eh5.shape)


		return [ecfe, ec1, ec2, ec3, ec4, self.encode_ct(ec5)], [[], [], [], [], [], self.encode_ht(eh5)]



##########################################################
class DRUNetCAMDecoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer = 2, fuse_model='csfm', norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetCAMDecoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.fuse_model = fuse_model
		self.hl = haze_layer
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		n_downsampling = 3
		
		mult = 32
		#sft_5 = [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias)]
		#self.sft_5 = nn.Sequential(*sft_5)

		
		decode_ft = []
		for i in range(2):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		#decode_ft += [CAM(ngf * mult, 16)]
		self.decode_ft = nn.Sequential(*decode_ft)


		mult = 2 * 32
		#decoder_hf5 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		decoder_hf5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf5 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_hf5 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf5 = nn.Sequential(*decoder_hf5)

		#decoder_hf4 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]

		mult = 2 * 16
		decoder_hf4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf4 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_hf4 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf4 = nn.Sequential(*decoder_hf4)

		#decoder_hf3 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		mult = 2 * 8
		decoder_hf3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf3 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_hf3 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf3 = nn.Sequential(*decoder_hf3)

		mult = 2 * 4
		decoder_hf2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf2 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		decoder_hf2 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf2 = nn.Sequential(*decoder_hf2)

		mult = 2 * 2
		decoder_hf1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)]
		decoder_hf1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#decoder_hf1 += [CAM(int(ngf * mult/4), 16)]
		self.decoder_hf1 = nn.Sequential(*decoder_hf1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#skip_fn = [nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)
		#skip_fe = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fe = nn.Sequential(*skip_fe)
		#skip_1 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 2), nn.ReLU(True)]
		#self.skip_1 = nn.Sequential(*skip_1)
		#skip_2 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 4), nn.ReLU(True)]
		#self.skip_2 = nn.Sequential(*skip_2)
		#skip_3 = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 8), nn.ReLU(True)]
		#self.skip_3 = nn.Sequential(*skip_3)
	
	def forward(self, input, c, h):
		
		
		fuse_5 = c[5]
		fuse_5 = adain(fuse_5, h[5])
		#fuse_5 = self.sft_5(torch.cat((c[5], h[5]), 1))


		dft = self.decode_ft(c[5])
		#print('fuse_5', fuse_5.shape, 'dft', dft.shape)
		concat = torch.cat((fuse_5, dft), 1)
		dhf5 = self.decoder_hf5(concat)

		#print('fuse_4', fuse_4.shape, 'dhf5', dhf5.shape)
		concat = torch.cat((c[4], dhf5), 1)
		dhf4 = self.decoder_hf4(concat)

		#print('fuse_3', fuse_4.shape, 'dhf4', dhf5.shape)
		concat = torch.cat((c[3], dhf4), 1)
		dhf3 = self.decoder_hf3(concat)

		concat = torch.cat((c[2], dhf3), 1)
		dhf2 = self.decoder_hf2(concat)

		concat = torch.cat((c[1], dhf2), 1)
		dhf1 = self.decoder_hf1(concat)
		#print(c[0].shape, self.skip_fn(input).shape, dhf1.shape, dhf2.shape, dhf3.shape, dhf4.shape, dhf5.shape)

		concat = torch.cat((c[0], dhf1), 1)
		out = self.decoder_fn(concat) 
		return out





#---------------------------------------------------------



########################      #######################################
##############################################      #################
#####################################################################
##                             DR + UNet                           ##
##       Disentangled Reperesentation with U-Net Architecture      ##
#####################################################################
##############################################      #################
########################      #######################################

## Encoder for Clear Image ##

class DRUNetEncoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetEncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		#encode_contfe += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(ngf * mult *2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		#encode_cont1 += [CAM()]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		ec5 = self.encode_cont5(ec4)
		#print('ec5', ec5.shape)
		return [ecfe, ec1, ec2, ec3, ec4, ec5]




##########################################################
## Encoder for Clear image 
##########################################################
class DRUNetDecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetDecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		n_downsampling = 3
		mult = 32
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)

		mult = 2 * 32
		decoder_up5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up5 += [nn.Conv2d(int(ngf * mult /2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up5 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up5 = nn.Sequential(*decoder_up5)

		mult = 2 * 16
		decoder_up4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up4 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up4 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up4 = nn.Sequential(*decoder_up4)

		mult = 2 * 8
		decoder_up3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up3 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up3 = nn.Sequential(*decoder_up3)

		mult = 2 * 4
		decoder_up2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up2 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up2 = nn.Sequential(*decoder_up2)

		mult = 2 * 2
		decoder_up1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up1 = nn.Sequential(*decoder_up1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]  #+ int(ngf/2)
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]  #+ int(ngf/2)
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#
		#skip_fn = [nn.Conv2d(3, int(ngf), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)


		#self.dehaze = Dehaze()

	
	def forward(self, input, c):
		dft = self.decode_ft(c[5])

		concat = torch.cat((c[5], dft), 1) # self.skip_3(c[3])
		dup5 = self.decoder_up5(concat)

		concat = torch.cat((c[4], dup5), 1)  # self.skip_2(c[2])
		dup4 = self.decoder_up4(concat)

		concat = torch.cat((c[3], dup4), 1) # self.skip_3(c[3])
		dup3 = self.decoder_up3(concat)

		concat = torch.cat((c[2], dup3), 1)  # self.skip_2(c[2])
		dup2 = self.decoder_up2(concat)

		concat = torch.cat((c[1], dup2), 1)  # self.skip_1(c[1])
		dup1 = self.decoder_up1(concat)

		concat = torch.cat((c[0], dup1), 1) # self.skip_fe(c[0]), self.skip_fn(input)
		#print(concat.shape)
		out = self.decoder_fn(concat)

		return out#, dehaze


########################################################################


class DRUNetEncoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer=2, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, ca_type='cross_ca', padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetEncoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.hl = haze_layer
		self.ca_type = ca_type
		self.n_blocks = n_blocks
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		#self.sa = SpatialAttention(7)

		## Content Encoder ##
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		##################
		## Haze Encoder ##
		##################
		encode_hazefe = [nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_hazefe = nn.Sequential(*encode_hazefe)

		# hazy style layer 1 #
		mult = 1
		encode_haze1 = [nn.AvgPool2d(2, stride=2)]
		
		encode_haze1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		self.encode_haze1 = nn.Sequential(*encode_haze1)


		# hazy style layer 2 #
		mult = 2
		encode_haze2 = [nn.AvgPool2d(2, stride=2)]
		ratio = 8
		encode_haze2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		self.encode_haze2 = nn.Sequential(*encode_haze2)
		#self.channel_att_2 = ChannelAttention(int(ngf * mult * 2), int(ngf * mult * 2/ratio))

		## 
		mult = 4
		
		haze_feature_l1 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		mult = 8
		
		haze_feature_l2 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		mult = 16
		
		haze_feature_l3 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		self.haze_feature_l1 = nn.Sequential(*haze_feature_l1)
		self.haze_feature_l2 = nn.Sequential(*haze_feature_l2)
		self.haze_feature_l3 = nn.Sequential(*haze_feature_l3)

		



	def forward(self, input):
		ecfe = self.encode_contfe(input)
		#print('encoder-hazy: input', input.shape)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		#print('encoder-hazy: ec4', ec4.shape)
		ec5 = self.encode_cont5(ec4)
		#print('encoder-hazy: ec5', ec5.shape)

		ehfe = self.encode_hazefe(input)

		eh1 = self.encode_haze1(ehfe)
		eh2 = self.encode_haze2(eh1)
		eh3 = self.haze_feature_l1(eh2)
		eh4 = self.haze_feature_l2(eh3)
		eh5 = self.haze_feature_l3(eh4)

		#print('encoder-hazy: eh5+conv.', eh5.shape)


		


		return [ecfe, ec1, ec2, ec3, ec4, ec5], [[], [], [], [], [], eh5]



##########################################################
class DRUNetDecoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer = 2, fuse_model='csfm', norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRUNetDecoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.fuse_model = fuse_model
		self.hl = haze_layer
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		n_downsampling = 3
		
		mult = 32
		sft_5 = [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias)]
		self.sft_5 = nn.Sequential(*sft_5)

		
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)


		mult = 2 * 32
		#decoder_hf5 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		decoder_hf5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf5 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf5 = nn.Sequential(*decoder_hf5)

		#decoder_hf4 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]

		mult = 2 * 16
		decoder_hf4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf4 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf4 = nn.Sequential(*decoder_hf4)

		#decoder_hf3 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		mult = 2 * 8
		decoder_hf3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf3 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf3 = nn.Sequential(*decoder_hf3)

		mult = 2 * 4
		decoder_hf2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/4), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf2 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf2 = nn.Sequential(*decoder_hf2)

		mult = 2 * 2
		decoder_hf1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/4), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/4)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf1 += [ResnetBlockDec(int(ngf * mult/4), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf1 = nn.Sequential(*decoder_hf1)

		mult = 2
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#skip_fn = [nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)
		#skip_fe = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fe = nn.Sequential(*skip_fe)
		#skip_1 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 2), nn.ReLU(True)]
		#self.skip_1 = nn.Sequential(*skip_1)
		#skip_2 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 4), nn.ReLU(True)]
		#self.skip_2 = nn.Sequential(*skip_2)
		#skip_3 = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 8), nn.ReLU(True)]
		#self.skip_3 = nn.Sequential(*skip_3)
	
	def forward(self, input, c, h):
		
		
		fuse_5 = c[5]
		fuse_5 = adain(fuse_5, h[5])
		#fuse_5 = self.sft_5(torch.cat((c[5], h[5]), 1))


		dft = self.decode_ft(c[5])
		#print('fuse_5', fuse_5.shape, 'dft', dft.shape)
		concat = torch.cat((fuse_5, dft), 1)
		dhf5 = self.decoder_hf5(concat)

		#print('fuse_4', fuse_4.shape, 'dhf5', dhf5.shape)
		concat = torch.cat((c[4], dhf5), 1)
		dhf4 = self.decoder_hf4(concat)

		#print('fuse_3', fuse_4.shape, 'dhf4', dhf5.shape)
		concat = torch.cat((c[3], dhf4), 1)
		dhf3 = self.decoder_hf3(concat)

		concat = torch.cat((c[2], dhf3), 1)
		dhf2 = self.decoder_hf2(concat)

		concat = torch.cat((c[1], dhf2), 1)
		dhf1 = self.decoder_hf1(concat)
		#print(c[0].shape, self.skip_fn(input).shape, dhf1.shape, dhf2.shape, dhf3.shape, dhf4.shape, dhf5.shape)

		concat = torch.cat((c[0], dhf1), 1)
		out = self.decoder_fn(concat) 
		return out






########################      #######################################
##############################################      #################
#####################################################################
##                                 DR                              ##
##     Disentangled Reperesentation without U-Net Architecture     ##
#####################################################################
##############################################      #################
########################      #######################################

## Encoder for Clear Image ##

class DREncoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DREncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		#encode_contfe += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(ngf * mult *2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(ngf * mult*2, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		ec5 = self.encode_cont5(ec4)
		#print('ec5', ec5.shape)
		return [[], [], [], [], [], ec5]




##########################################################
## Encoder for Clear image 
##########################################################
class DRDecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRDecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer
		n_downsampling = 3
		mult = 32
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)

		mult = 2 * 16
		decoder_up5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up5 += [nn.Conv2d(int(ngf * mult /2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up5 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up5 = nn.Sequential(*decoder_up5)

		mult = 2 * 8
		decoder_up4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		#decoder_up4 += [nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		decoder_up4 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up4 = nn.Sequential(*decoder_up4)

		mult = 2 * 4
		decoder_up3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up3 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up3 = nn.Sequential(*decoder_up3)

		mult = 2 * 2
		decoder_up2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up2 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up2 = nn.Sequential(*decoder_up2)

		mult = 2
		decoder_up1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_up1 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_up1 = nn.Sequential(*decoder_up1)

		mult = 1
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]  #+ int(ngf/2)
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]  #+ int(ngf/2)
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#
		#skip_fn = [nn.Conv2d(3, int(ngf), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)


		#self.dehaze = Dehaze()

	
	def forward(self, input, c):
		dft = self.decode_ft(c[5])

		#concat = torch.cat((c[5], dft), 1) # self.skip_3(c[3])
		dup5 = self.decoder_up5(dft)

		#concat = torch.cat((c[4], dup5), 1)  # self.skip_2(c[2])
		dup4 = self.decoder_up4(dup5)

		#concat = torch.cat((c[3], dup4), 1) # self.skip_3(c[3])
		dup3 = self.decoder_up3(dup4)

		#concat = torch.cat((c[2], dup3), 1)  # self.skip_2(c[2])
		dup2 = self.decoder_up2(dup3)

		#concat = torch.cat((c[1], dup2), 1)  # self.skip_1(c[1])
		dup1 = self.decoder_up1(dup2)

		#concat = torch.cat((c[0], dup1), 1) # self.skip_fe(c[0]), self.skip_fn(input)
		#print(concat.shape)
		out = self.decoder_fn(dup1)

		return out#, dehaze


########################################################################


class DREncoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer=2, norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, ca_type='cross_ca', padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DREncoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.hl = haze_layer
		self.ca_type = ca_type
		self.n_blocks = n_blocks
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		#self.sa = SpatialAttention(7)

		## Content Encoder ##
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont1 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont2 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont3 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		mult = 8
		encode_cont4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont4 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont4 = nn.Sequential(*encode_cont4)

		mult = 16
		encode_cont5 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		encode_cont5 += [ResnetBlockEnc(int(ngf * mult*2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.encode_cont5 = nn.Sequential(*encode_cont5)

		##################
		## Haze Encoder ##
		##################
		encode_hazefe = [nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_hazefe = nn.Sequential(*encode_hazefe)

		# hazy style layer 1 #
		mult = 1
		encode_haze1 = [nn.AvgPool2d(2, stride=2)]
		
		encode_haze1 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		self.encode_haze1 = nn.Sequential(*encode_haze1)


		# hazy style layer 2 #
		mult = 2
		encode_haze2 = [nn.AvgPool2d(2, stride=2)]
		ratio = 8
		encode_haze2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
						norm_layer(ngf * mult * 2),
						nn.LeakyReLU(0.2, True)]

		self.encode_haze2 = nn.Sequential(*encode_haze2)
		#self.channel_att_2 = ChannelAttention(int(ngf * mult * 2), int(ngf * mult * 2/ratio))

		## 
		mult = 4
		
		haze_feature_l1 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		mult = 8
		
		haze_feature_l2 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		mult = 16
		
		haze_feature_l3 = [nn.AvgPool2d(2, stride=2), nn.Conv2d(int(ngf * mult), int(ngf * mult * 2), kernel_size=3, stride=1, padding=1), norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, True)]

		self.haze_feature_l1 = nn.Sequential(*haze_feature_l1)
		self.haze_feature_l2 = nn.Sequential(*haze_feature_l2)
		self.haze_feature_l3 = nn.Sequential(*haze_feature_l3)

		



	def forward(self, input):
		ecfe = self.encode_contfe(input)
		#print('encoder-hazy: input', input.shape)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		ec4 = self.encode_cont4(ec3)
		#print('encoder-hazy: ec4', ec4.shape)
		ec5 = self.encode_cont5(ec4)
		#print('encoder-hazy: ec5', ec5.shape)

		ehfe = self.encode_hazefe(input)

		eh1 = self.encode_haze1(ehfe)
		eh2 = self.encode_haze2(eh1)
		eh3 = self.haze_feature_l1(eh2)
		eh4 = self.haze_feature_l2(eh3)
		eh5 = self.haze_feature_l3(eh4)

		#print('encoder-hazy: eh5+conv.', eh5.shape)


		


		return [[], [], [], [], [], ec5], [[], [], [], [], [], eh5]



##########################################################
class DRDecoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer = 2, fuse_model='csfm', norm_layer=ActNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(DRDecoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.fuse_model = fuse_model
		self.hl = haze_layer
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == norm_layer
		else:
			use_bias = norm_layer == norm_layer

		n_downsampling = 3
		
		mult = 32
		#sft_5 = [nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias)]
		#self.sft_5 = nn.Sequential(*sft_5)

		
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlockDec(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)


		mult = 2 * 16
		#decoder_hf5 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		decoder_hf5 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf5 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf5 = nn.Sequential(*decoder_hf5)

		#decoder_hf4 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]

		mult = 2 * 8
		decoder_hf4 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf4 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf4 = nn.Sequential(*decoder_hf4)

		#decoder_hf3 = [nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ngf * mult + ngf, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult)), nn.ReLU(True)]
		mult = 2 * 4
		decoder_hf3 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf3 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf3 = nn.Sequential(*decoder_hf3)

		mult = 2 * 2
		decoder_hf2 = [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult/2), kernel_size=4, stride=2, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True)]#[nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf2 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf2 = nn.Sequential(*decoder_hf2)

		mult = 2 
		decoder_hf1 = [nn.Conv2d(int(ngf * mult), int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias),norm_layer(int(ngf * mult/2)), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear')]
		decoder_hf1 += [ResnetBlockDec(int(ngf * mult/2), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
		self.decoder_hf1 = nn.Sequential(*decoder_hf1)

		mult = 1
		decoder_fn = [nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult)), nn.ReLU(True)]
		#decoder_fn += [nn.Conv2d(ngf * mult, int(ngf * mult/2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult/2)), nn.ReLU(True)]
		decoder_fn += [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)

		#skip_fn = [nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fn = nn.Sequential(*skip_fn)
		#skip_fe = [nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
		#self.skip_fe = nn.Sequential(*skip_fe)
		#skip_1 = [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 2), nn.ReLU(True)]
		#self.skip_1 = nn.Sequential(*skip_1)
		#skip_2 = [nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 4), nn.ReLU(True)]
		#self.skip_2 = nn.Sequential(*skip_2)
		#skip_3 = [nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf * 8), nn.ReLU(True)]
		#self.skip_3 = nn.Sequential(*skip_3)
	
	def forward(self, input, c, h):
		
		
		fuse_5 = c[5]
		fuse_5 = adain(fuse_5, h[5])
		#fuse_5 = self.sft_5(torch.cat((c[5], h[5]), 1))


		dft = self.decode_ft(c[5])
		#print('fuse_5', fuse_5.shape, 'dft', dft.shape)
		#concat = torch.cat((fuse_5, dft), 1)
		dhf5 = self.decoder_hf5(dft)

		#print('fuse_4', fuse_4.shape, 'dhf5', dhf5.shape)
		#concat = torch.cat((c[4], dhf5), 1)
		dhf4 = self.decoder_hf4(dhf5)

		#print('fuse_3', fuse_4.shape, 'dhf4', dhf5.shape)
		#concat = torch.cat((c[3], dhf4), 1)
		dhf3 = self.decoder_hf3(dhf4)

		#concat = torch.cat((c[2], dhf3), 1)
		dhf2 = self.decoder_hf2(dhf3)

		#concat = torch.cat((c[1], dhf2), 1)
		dhf1 = self.decoder_hf1(dhf2)
		#print(c[0].shape, self.skip_fn(input).shape, dhf1.shape, dhf2.shape, dhf3.shape, dhf4.shape, dhf5.shape)

		#concat = torch.cat((c[0], dhf1), 1)
		out = self.decoder_fn(dhf1) 
		return out


##########################################################################




# Define a resnet block
class ResnetBlockDec(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlockDec, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

		self.relu = nn.ReLU(True)
	
	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               norm_layer(dim),
		               nn.ReLU(True)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               norm_layer(dim)]
		
		return nn.Sequential(*conv_block)

	
	
	def forward(self, x):
		out = self.relu(x + self.conv_block(x))
		return out

class ResnetBlockEnc(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlockEnc, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

		self.relu = nn.LeakyReLU(0.2, True)
	
	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               norm_layer(dim),
		               nn.LeakyReLU(0.2, True)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               norm_layer(dim)]
		
		return nn.Sequential(*conv_block)

	
	
	def forward(self, x):
		out = self.relu(x + self.conv_block(x))
		return out






# Define a resnet block
class ResnetBlockDecNN(nn.Module):
	def __init__(self, dim, padding_type, use_dropout, use_bias):
		super(ResnetBlockDecNN, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

		self.relu = nn.ReLU(True)
	
	def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               nn.ReLU(True)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
		
		return nn.Sequential(*conv_block)

	
	
	def forward(self, x):
		out = self.relu(x + self.conv_block(x))
		return out

class ResnetBlockEncNN(nn.Module):
	def __init__(self, dim, padding_type, use_dropout, use_bias):
		super(ResnetBlockEncNN, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

		self.relu = nn.LeakyReLU(0.2, True)
	
	def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               nn.LeakyReLU(0.2, True)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
		
		return nn.Sequential(*conv_block)

	
	
	def forward(self, x):
		out = self.relu(x + self.conv_block(x))
		return out








'''
# Defines the UNet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UNetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64,
	             norm_layer=nn.BatchNorm2d, use_dropout=False):
		super(UNetGenerator, self).__init__()
		
		# construct unet structure
		unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
		for i in range(num_downs - 5):
			unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
		
		self.model = unet_block
	
	def forward(self, input):
		return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UNetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc, input_nc=None,
	             submodule=None, outermost=False, innermost=False, norm_layer=ActNorm2d, use_dropout=False):
		super(UNetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == ActNorm2d
		else:
			use_bias = norm_layer == ActNorm2d
		if input_nc is None:
			input_nc = outer_nc
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
		                     stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)
		
		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]
			
			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up
		
		self.model = nn.Sequential(*model)
	
	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([x, self.model(x)], 1)

'''
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerFeatureDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(NLayerFeatureDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == ActNorm2d
		else:
			use_bias = norm_layer == ActNorm2d
		
		kw = 4
		padw = 1
		rf_e = [nn.ReflectionPad2d(3),
			nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1),
			nn.LeakyReLU(0.2, True)]

		rf_e += [ nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
			]

		#rf_2 = [ nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 4),
		#		nn.LeakyReLU(0.2, True)
		#	]

		#rf_3 = [ nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 8),
		#		nn.LeakyReLU(0.2, True)
		#	]
		
		self.rf_e = nn.Sequential(*rf_e)
		#self.rf_1 = nn.Sequential(*rf_1)
		#self.rf_2 = nn.Sequential(*rf_2)
		#self.rf_3 = nn.Sequential(*rf_3)

		ff_e = [nn.ReflectionPad2d(3),
			nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1),
			nn.LeakyReLU(0.2, True)]

		ff_e += [ nn.Conv2d(ndf, ndf * 2, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
			]

		#ff_2 = [ nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 4),
		#		nn.LeakyReLU(0.2, True)
		#	]

		#ff_3 = [ nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 8),
		#		nn.LeakyReLU(0.2, True)
		#	]
		
		self.ff_e = nn.Sequential(*ff_e)
		#self.ff_1 = nn.Sequential(*ff_1)
		#self.ff_2 = nn.Sequential(*ff_2)
		#self.ff_3 = nn.Sequential(*ff_3)

		fuse1 = [nn.Conv2d(ndf * 4, ndf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
				]
		self.fuse1 = nn.Sequential(*fuse1)

		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		reduction = 16
		d = max(int(ndf * 2/reduction),4)
		self.conv_squeeze = nn.Sequential(nn.Conv2d(ndf * 2, d, 1, padding=0, bias=use_bias), nn.PReLU())
		self.fcs_f0 = nn.Conv2d(d, ndf * 2, kernel_size=1, stride=1,bias=use_bias)
		self.fcs_f1 = nn.Conv2d(d, ndf * 2, kernel_size=1, stride=1,bias=use_bias)
		self.softmax = nn.Softmax(dim=2)

		fuse2 = [
				nn.Conv2d(ndf * 2 , ndf * 2,
				          kernel_size=3, stride=1, padding=padw, bias=use_bias),
				norm_layer(ndf * 2),
				nn.LeakyReLU(0.2, True)
			]
		self.fuse2 = nn.Sequential(*fuse2)

		cross_f_1 = [
				nn.Conv2d(ndf * 2 , ndf * 4,
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 4),
				nn.LeakyReLU(0.2, True)
			]

		cross_f_2 = [
				nn.Conv2d(ndf * 4, ndf * 8, 
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * 8),
				nn.LeakyReLU(0.2, True)
			]

		cross_f_3 = [
				nn.Conv2d(ndf * 8, ndf * 8,
				          kernel_size=3, stride=1, padding=1, bias=use_bias),
				norm_layer(ndf * 8),
				nn.LeakyReLU(0.2, True)
			]

		#cross_f_5 = [
		#		nn.Conv2d(ndf * 8, ndf * 8,
		#		          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
		#		norm_layer(ndf * 8),
		#		nn.LeakyReLU(0.2, True)
		#	]

		#cross_att = [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=(4,8), stride=1, padding=0)]

		self.cross_f_1 = nn.Sequential(*cross_f_1)
		self.cross_f_2 = nn.Sequential(*cross_f_2)
		self.cross_f_3 = nn.Sequential(*cross_f_3)
		#self.cross_f_4 = nn.Sequential(*cross_f_4)
		#self.cross_f_5 = nn.Sequential(*cross_f_5)
		#self.cross_att = nn.Sequential(*cross_att)

		class_3 = [nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)]
		#class_4 = [nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=padw)]
		#class_5 = [nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=padw)]

		self.class_3 = nn.Sequential(*class_3)
		#self.class_4 = nn.Sequential(*class_4)
		#self.class_5 = nn.Sequential(*class_5)

		
		sigm = [nn.Sigmoid()]
		
		self.sigm = nn.Sequential(*sigm)


	def forward(self, real, fake):
		real_fe = self.rf_e(real)
		#real_f1 = self.rf_1(real_fe)
		#real_f2 = self.rf_2(real_f1)
		#real_f3 = self.rf_3(real_f2)

		fake_fe = self.ff_e(fake)
		#fake_f1 = self.ff_1(fake_fe)
		#fake_f2 = self.ff_2(fake_f1)
		#fake_f3 = self.ff_3(fake_f2)

		rf_fuse = self.fuse1(torch.cat((real_fe, fake_fe),1))
		pooling = self.global_avg_pool(rf_fuse)
		squeeze = self.conv_squeeze(pooling)
		score_f0 = self.fcs_f0(squeeze)
		score_f1 = self.fcs_f1(squeeze)
		score_cat = torch.cat((score_f0, score_f1),2)
		score_att = self.softmax(score_cat)
		score_chunk = torch.chunk(score_att, 4, 2)
		real_fe = score_chunk[0] * real_fe
		fake_fe = score_chunk[1] * fake_fe
		rf_fuse = self.fuse2(real_fe + fake_fe)

		cf1 = self.cross_f_1(rf_fuse)
		cf2 = self.cross_f_2(cf1)
		cf3 = self.cross_f_3(cf2)
		#cf4 = self.cross_f_4(cf3)
		#cf5 = self.cross_f_5(cf4)

		#cca = self.sigm(self.cross_att(cf3))

		#print('cf3', cf3.shape, 'cf4', cf4.shape, 'cca', cca.shape)

		cls3 = self.class_3(cf3)
		#cls4 = self.class_4(cca * cf4 + cf4)
		#cls5 = self.class_4(cca * cf5 + cf5)
		#print(cls3.shape, cls4.shape,cls5.shape,)

		out = self.sigm(cls3)
		#print(out.shape)
		return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(NLayerDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == ActNorm2d
		else:
			use_bias = norm_layer == ActNorm2d
		
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
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]
		
		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
			          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		
		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
		
		if use_sigmoid:
			sequence += [nn.Sigmoid()]
		
		self.model = nn.Sequential(*sequence)
	
	def forward(self, input):
		return self.model(input)

class PixelDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(PixelDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == ActNorm2d
		else:
			use_bias = norm_layer == ActNorm2d
		
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
	def __init__(self, input_nc, ndf, norm_layer=nn.BatchNorm2d):
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
				norm_layer(ndf * nf_mult),
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
