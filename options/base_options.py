import argparse
import os

import torch
from util import util


class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False
	
	def initialize(self):
		self.parser.add_argument('--dataroot', required=True, default='D:/dzhao/DATASET/Mars/MarsGAN/Mars_mix/',help='path to images (should have subfolders trainA, trainB, etc)')
		self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
		self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
		self.parser.add_argument('--fineSize', type=int, default=128, help='then crop to this size')
		self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
		self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
		self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
		self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
		self.parser.add_argument('--which_model_netD', type=str, default='nl', help='selects model to use for netD')
		self.parser.add_argument('--which_model_netG', type=str, default='dr', help='selects model to use for netG')
		self.parser.add_argument('--ca_type', type=str, default='cross_ca', help='disentangling the haze representations using which channel attention model: cross channel attenion [cross_ca]; channal attention for each level [level_ca]; or no channel attention [none_ca]')
		self.parser.add_argument('--fuse_model', type=str, default='cat', help='selects model to fuse the content and hazy style features, cat|csfm')
		self.parser.add_argument('--hl', type=int, default=1, help=' layers of the Hierarchy DR')
		self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
		self.parser.add_argument('--unet_layer', type=int, default=3, help='only used if which_model_netD==n_layers')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
		
		self.parser.add_argument('--dataset_mode', type=str, default='dr_uspv', help='chooses how datasets are loaded:dh_hdr_sspv, dh_hdr_uspv.')
		self.parser.add_argument('--model', type=str, default='dr_uspv', help='chooses which model to use. ')
		self.parser.add_argument('--weights_model_type', type=str, default='drn26', help='chooses which model to use. drn26, fcn8s')
		self.parser.add_argument('--dh_real', action='store_true', help='if true, dehaze the real-world hazy images (unsupervised model)')
		self.parser.add_argument('--multisource', action='store_true', help='if true, dehaze the real-world hazy images (unsupervised model)')
		self.parser.add_argument('--max_epoch', default=300, type=int)
		self.parser.add_argument('--current_epoch', default=0, type=int)
		self.parser.add_argument('--weights_init', type=str)
		self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
		self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
		self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		self.parser.add_argument('--norm', type=str, default='sswitch', help='the normalization | batch | instance | actnorm | switch | ssn |')
		self.parser.add_argument('--sb', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
		self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
		self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
		self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
		self.parser.add_argument('--display_port', type=int, default=3000, help='visdom port of the web display')
		self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
		self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, ' 'only a subset is loaded.')
		self.parser.add_argument('--resize_or_crop', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
		self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
		self.parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming|orthogonal]')
		self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
		self.parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
		self.parser.add_argument('--out_all', action='store_true', help='output all stylized images(fake_B_{})')
		self.parser.add_argument('--allmodel', action='store_true', help='output all stylized images(fake_B_{})')
		self.parser.add_argument('--depth', action='store_true', help='using depth map')
		self.parser.add_argument('--DHN_frozen_epoch', type=int, default=60)
		self.parser.add_argument('--Shared_DT', action='store_true', help="Shared target domain generator")
		self.parser.add_argument('--model_type', type=str, default='fcn8s', help="choose to load which type of model (fcn8s, drn26, deeplabv2)")
		self.parser.add_argument('--dehazing_loss', action='store_true', help='use dehazing loss')
		self.parser.add_argument('--general_dehazing_weight', type=float, default=0.2, help='weight for dehazing loss')
		self.parser.add_argument('--weights_clear', type=str, default='clear', help='init weights for clear dataset')
		self.parser.add_argument('--weights_hazy', type=str, default='hazy', help='init weights for hazy dataset')
		self.parser.add_argument('--pretrained_model', type=str, default='pix2pixHD', help='init weights for O-HAZE dataset')
		self.parser.add_argument('--dehazing_netG', type=str, default='local', help='selects model to use for netG')
		self.parser.add_argument('--epdn_ngf', type=int, default=32, help='# of gen filters in first conv layer')
		self.parser.add_argument('--n_downsample_global', type=int, default=2, help='number of downsampling layers in netG')
		self.parser.add_argument('--n_blocks', type=int, default=2, help='number of residual blocks in the global generator network')

		self.parser.add_argument('--num_D', type=int, default=3, help='number of discriminators to use')

		self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
		self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')
		self.parser.add_argument('--gradient_loss', action='store_true', help='use gradient_loss')
		self.parser.add_argument('--gradient_dehaze_loss', action='store_true', help='use gradient_loss')
		self.parser.add_argument('--imagepool', action='store_true', help='use the image pool')

		# opt of the pretarined model
		self.parser.add_argument('--pre_epdn_ngf', type=int, default=16, help='# of gen filters in first conv layer')
		self.parser.add_argument('--pre_n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
		self.parser.add_argument('--pre_n_blocks_global', type=int, default=3, help='number of residual blocks in the global generator network')
		self.parser.add_argument('--pre_n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
		self.parser.add_argument('--pre_n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
		self.parser.add_argument('--pre_netG', type=str, default='local', help='selects model to use for netG')

		self.parser.add_argument('--inference_script', type=str, default='', help='inference script')
		self.parser.add_argument('--dynamic_weight', type=float, default=10, help='Weight for Dynamic Dehazing Loss(KL div) loss')
		self.initialized = True
	
	def parse(self):
		if not self.initialized:
			self.initialize()
		opt = self.parser.parse_args()
		opt.isTrain = self.isTrain  # train or test
		
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		
		# set gpu ids
		if len(opt.gpu_ids) > 0:
			torch.cuda.set_device(opt.gpu_ids[0])
		
		args = vars(opt)
		
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')
		
		if opt.suffix:
			suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
			opt.name = opt.name + suffix
		# save to the disk
		expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		self.opt = opt
		return self.opt
