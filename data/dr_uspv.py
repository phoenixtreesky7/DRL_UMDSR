import os.path
import random

import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
#from data.cityscapes import remap_labels_to_train_ids
from data.image_folder import make_cs_labels, make_dataset

# This dataset is used to conduct double cyclegan for both GTAV->CityScapes and Synthia->CityScapes
class DR_USPV(BaseDataset):
	def initialize(self, opt):
		# OHAZE as dataset 1
		# 3D60 as dataset 2
		self.opt = opt
		self.root = opt.dataroot
		if self.opt.allmodel:
			self.dir_A = os.path.join(opt.dataroot, 'trainA')  # real_clear  clear/resize
			self.A_paths = make_dataset(self.dir_A)
			self.A_paths = sorted(self.A_paths)
			self.A_size = len(self.A_paths)
		
		
		self.dir_B = os.path.join(opt.dataroot, 'trainB')   # real_hazy   hazy/resize
		
		self.B_paths = make_dataset(self.dir_B)
		
		self.B_paths = sorted(self.B_paths)

		self.B_size = len(self.B_paths)

		self.transform = get_transform(opt)

	
	def __getitem__(self, index):
		index_B = index % self.B_size
		B_path = self.B_paths[index_B]
		#B_path = self.B_paths[index % self.B_size]
		
		if self.opt.allmodel:
			if self.opt.sb:
				index_A = index % self.A_size
			else:
				index_A = random.randint(0, self.A_size - 1)

			A_path = self.A_paths[index_A]
			#if self.opt.sb:
			#	index_C = index % self.A_size
			#else:
			#	index_C = random.randint(0, self.A_size - 1)
			#C_path = self.A_paths[index_C]

			#if self.opt.sb:
			#	index_D = index % self.B_size
			#else:
			#	index_D = random.randint(0, self.B_size - 1)
			#D_path = self.B_paths[index_D]

		if self.opt.allmodel:
			A_img = Image.open(A_path).convert('RGB')
			
			#C_img = Image.open(C_path).convert('RGB')
			#D_img = Image.open(D_path).convert('RGB')
			A = self.transform(A_img)
			
			#C = self.transform(C_img)
			#D = self.transform(D_img)

		B_img = Image.open(B_path).convert('RGB')
		
		B = self.transform(B_img)
		

		#if self.opt.which_direction == 'BtoA':
		#	input_nc = self.opt.output_nc
		#	output_nc = self.opt.input_nc
		#else:
		#	input_nc = self.opt.input_nc
		#	output_nc = self.opt.output_nc
		
		#if self.opt.isTrain:
		#	if input_nc == 1:  # RGB to gray
		#		tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
		#		A = tmp.unsqueeze(0)


		#if output_nc == 1:  # RGB to gray
		#	tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
		#	B = tmp.unsqueeze(0)
		
		if self.opt.allmodel:
			return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
		else:
			return {'B': B, 'B_paths': B_path}
		

	def __len__(self):
		return self.B_size
	
	def name(self):
		return 'DR_USPV'
