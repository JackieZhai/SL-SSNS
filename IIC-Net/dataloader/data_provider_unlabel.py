from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import h5py
import math
import time
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio 
from utils.consistency_aug_perturbations import Rescale
from utils.augmentation import SimpleAugment as Filp 
from utils.consistency_aug_perturbations import Intensity
from utils.consistency_aug_perturbations import GaussBlur
from utils.consistency_aug_perturbations import GaussNoise
from utils.consistency_aug_perturbations import Cutout
from utils.consistency_aug_perturbations import SobelFilter
from utils.consistency_aug_perturbations import Mixup
from utils.consistency_aug_perturbations import Elastic
from utils.consistency_aug_perturbations import Artifact
from utils.consistency_aug_perturbations import Missing
from utils.consistency_aug_perturbations import BlurEnhanced


class Train(Dataset):
	def __init__(self, cfg):
		super(Train, self).__init__()
		self.cfg = cfg
		self.model_type = cfg.MODEL.model_type
		self.per_mode = cfg.DATA.per_mode

		# basic settings
		self.crop_size = [18, 160, 160]
		self.net_padding = [0, 100, 100]
		
		# training dataset files (h5), may contain many datasets
		if cfg.DATA.unlabel_dataset ==  'cremi-bc-all':
			self.sub_path = 'cremi-r/all'
			self.train_datasets = ['cremiB-top.tif', 'cremiC-top.tif']
		# elif cfg.DATA.unlabel_dataset ==  'cremi-bc-all-p1':
		# 	self.sub_path = 'cremi-r/all'
		# 	self.train_datasets = ['cremiB-top.tif', 'cremiC-top.tif', 'cremiB+-top.tif', 'cremiC+-top.tif']
		# elif cfg.DATA.unlabel_dataset ==  'cremi-bc-all-p2':
		# 	self.sub_path = 'cremi-r/all'
		# 	self.train_datasets = ['cremiB-top.tif', 'cremiC-top.tif''cremiB+-bottom.tif', 'cremiC+-bottom.tif', 'cremiB+-top.tif', 'cremiC+-top.tif']
		elif cfg.DATA.unlabel_dataset ==  'cremi-bc-all-p1':
			self.sub_path = 'cremi-r/all'
			self.train_datasets = ['cremiB+-top.tif', 'cremiC+-top.tif']
		elif cfg.DATA.unlabel_dataset ==  'cremi-bc-all-p2':
			self.sub_path = 'cremi-r/all'
			self.train_datasets = ['cremiB+-bottom.tif', 'cremiC+-bottom.tif']
		elif cfg.DATA.unlabel_dataset ==  'cremi-bc-all-p3':
			self.sub_path = 'cremi-r/all'
			self.train_datasets = ['cremiB+-bottom.tif', 'cremiC+-top.tif']
		else:
			raise AttributeError('No this dataset type!')

		# the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
		self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)

		# split unlabeled data

		# augmentation
		self.if_norm_images = cfg.DATA.if_norm_images
		self.if_scale_aug = cfg.DATA.if_scale_aug_unlabel
		self.scale_factor = cfg.DATA.scale_factor
		self.if_filp_aug = cfg.DATA.if_filp_aug_unlabel
		self.if_rotation_aug = cfg.DATA.if_rotation_aug_unlabel
		self.if_intensity_aug = cfg.DATA.if_intensity_aug_unlabel
		self.if_elastic_aug = cfg.DATA.if_elastic_aug_unlabel
		self.if_noise_aug = cfg.DATA.if_noise_aug_unlabel
		self.min_noise_std = cfg.DATA.min_noise_std
		self.max_noise_std = cfg.DATA.max_noise_std
		self.if_mask_aug = cfg.DATA.if_mask_aug_unlabel
		self.if_blur_aug = cfg.DATA.if_blur_aug_unlabel
		self.min_kernel_size = cfg.DATA.min_kernel_size
		self.max_kernel_size = cfg.DATA.max_kernel_size
		self.min_sigma = cfg.DATA.min_sigma
		self.max_sigma = cfg.DATA.max_sigma
		self.if_sobel_aug = cfg.DATA.if_sobel_aug_unlabel
		self.if_mixup_aug = cfg.DATA.if_mixup_aug_unlabel
		self.if_misalign_aug = cfg.DATA.if_misalign_aug_unlabel
		self.if_artifact_aug = cfg.DATA.if_artifact_aug_unlabel
		self.if_missing_aug = cfg.DATA.if_missing_aug_unlabel
		self.if_blurenhanced_aug = cfg.DATA.if_blurenhanced_aug_unlabel
		self.simple_aug = Filp()
		self.prob_u = cfg.TRAIN.mix_prob_u
		self.prob_u_m = cfg.TRAIN.mix_prob_u_m
		# load dataset
		self.dataset = []
		for k in range(len(self.train_datasets)):
			print('load unlabeled ' + self.train_datasets[k] + ' ...')
			# load raw data
			data = imageio.volread(os.path.join(self.folder_name, self.train_datasets[k]))
			self.dataset.append(data)
		
		for k in range(len(self.dataset)):
			if self.dataset[k].shape[0] == 65:
				self.dataset[k] = np.pad(self.dataset[k], ((self.net_padding[0], self.net_padding[0]),
														(self.net_padding[1], self.net_padding[1]),
														(self.net_padding[2], self.net_padding[2])), mode='reflect')
			else:
				self.dataset[k] = np.pad(self.dataset[k], ((2, 3),
														(self.net_padding[1], self.net_padding[1]),
														(self.net_padding[2], self.net_padding[2])), mode='reflect')

		# the training dataset size
		self.raw_data_shape = list(self.dataset[0].shape)
		print('unlabeld shape: ', self.raw_data_shape)

		# padding for augmentation
		self.sub_padding = [0, 80, 80]  # for rescale
		self.crop_from_origin = [self.crop_size[i] + 2*self.sub_padding[i] for i in range(len(self.sub_padding))]

		# perturbations
		self.perturbations_init()

	def __getitem__(self, index):
		# random select one dataset if contain many datasets
		k = random.randint(0, len(self.train_datasets)-1)
		used_data = self.dataset[k]

		# random select one sub-volume
		random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
		random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
		random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
		imgs1 = used_data[random_z:random_z+self.crop_from_origin[0], \
						random_y:random_y+self.crop_from_origin[1], \
						random_x:random_x+self.crop_from_origin[2]].copy()
		imgs1 = imgs1.astype(np.float32) / 255.0
		[imgs1] = self.simple_aug([imgs1])
		per_imgs1, scale_size1, rule1, rotnum1 = self.apply_perturbations(imgs1.copy(), mode=self.per_mode)
		gt_imgs1 = imgs1[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]

		if random.random() < self.prob_u:
			random_z = random.randint(0, self.raw_data_shape[0]-self.crop_from_origin[0])
			random_y = random.randint(0, self.raw_data_shape[1]-self.crop_from_origin[1])
			random_x = random.randint(0, self.raw_data_shape[2]-self.crop_from_origin[2])
			imgs2 = used_data[random_z:random_z+self.crop_from_origin[0], \
							random_y:random_y+self.crop_from_origin[1], \
							random_x:random_x+self.crop_from_origin[2]].copy()
			imgs2 = imgs2.astype(np.float32) / 255.0
			[imgs2] = self.simple_aug([imgs2])
			per_imgs2, scale_size2, rule2, rotnum2 = self.apply_perturbations(imgs2.copy(), mode=self.per_mode)
			gt_imgs2 = imgs2[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]

			random_y_mask = random.randint(0, 70)
			random_x_mask = random.randint(0, 70)
			mask = np.ones([18, 160, 160])
			mask[:,random_y_mask:random_y_mask+90, random_x_mask:random_x_mask+90] = 0

			# print(gt_imgs1.shape, gt_imgs2.shape)
			gt_imgs = mask * gt_imgs1 + (1-mask) * gt_imgs2
			per_imgs, scale_size, rule, rotnum = mask * per_imgs1 + (1-mask) * per_imgs2, scale_size1, rule1, rotnum1
		
		elif random.random() < self.prob_u_m:
			
			per_imgs = np.zeros_like(per_imgs1)
			gt_imgs = np.zeros_like(gt_imgs1)
			magic_order = [[0,80,0,80],[0,80,80,160],[80,160,0,80],[80,160,80,160]]
			magic_z = [1,1,1,1] # [random.choice([1,-1]),random.choice([1,-1]),random.choice([1,-1]),random.choice([1,-1])]
			random.shuffle(magic_order)
			per_imgs[:,0:80,0:80] = per_imgs1[::magic_z[0],magic_order[0][0]:magic_order[0][1],magic_order[0][2]:magic_order[0][3]]
			per_imgs[:,0:80,80:160] = per_imgs1[::magic_z[1],magic_order[1][0]:magic_order[1][1],magic_order[1][2]:magic_order[1][3]]
			per_imgs[:,80:160,0:80] = per_imgs1[::magic_z[2],magic_order[2][0]:magic_order[2][1],magic_order[2][2]:magic_order[2][3]]
			per_imgs[:,80:160,80:160] = per_imgs1[::magic_z[3],magic_order[3][0]:magic_order[3][1],magic_order[3][2]:magic_order[3][3]]

			gt_imgs[:,0:80,0:80] = gt_imgs1[::magic_z[0],magic_order[0][0]:magic_order[0][1],magic_order[0][2]:magic_order[0][3]]
			gt_imgs[:,0:80,80:160] = gt_imgs1[::magic_z[1],magic_order[1][0]:magic_order[1][1],magic_order[1][2]:magic_order[1][3]]
			gt_imgs[:,80:160,0:80] = gt_imgs1[::magic_z[2],magic_order[2][0]:magic_order[2][1],magic_order[2][2]:magic_order[2][3]]
			gt_imgs[:,80:160,80:160] = gt_imgs1[::magic_z[3],magic_order[3][0]:magic_order[3][1],magic_order[3][2]:magic_order[3][3]]
			
			scale_size, rule, rotnum = scale_size1, rule1, rotnum1
			
		else:
			per_imgs, scale_size, rule, rotnum = per_imgs1, scale_size1, rule1, rotnum1
			gt_imgs = gt_imgs1

		# extend dimension
		per_imgs = per_imgs[np.newaxis, ...]
		gt_imgs = gt_imgs[np.newaxis, ...]
		scale_size = np.asarray([scale_size], dtype=np.float32)
		scale_size = scale_size[np.newaxis, ...]
		rotnum = np.asarray([rotnum], dtype=np.float32)
		rotnum = rotnum[np.newaxis, ...]
		rule = rule.astype(np.float32)
		# rule = rule[np.newaxis, ...]
		per_imgs = np.ascontiguousarray(per_imgs, dtype=np.float32)
		gt_imgs = np.ascontiguousarray(gt_imgs, dtype=np.float32)

		return per_imgs, gt_imgs, scale_size, rule, rotnum

	def perturbations_init(self):
		self.per_rescale = Rescale(scale_factor=self.scale_factor, det_shape=self.crop_size)
		self.per_flip = Filp()
		self.per_intensity = Intensity()
		self.per_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
		self.per_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size, min_sigma=self.min_sigma, max_sigma=self.max_sigma)
		self.per_cutout = Cutout(model_type=self.model_type)
		self.per_sobel = SobelFilter(if_mean=True)
		self.per_mixup = Mixup(min_alpha=0.1, max_alpha=0.4)
		self.per_misalign = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 0, 0], prob_slip=0.2, prob_shift=0.2, max_misalign=17, padding=20)
		self.per_elastic = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 2, 2], padding=20)
		self.per_artifact = Artifact(min_sec=1, max_sec=5)
		self.per_missing = Missing(miss_fully_ratio=0.2, miss_part_ratio=0.5)
		self.per_blurenhanced = BlurEnhanced(blur_fully_ratio=0.5, blur_part_ratio=0.7)

	def apply_perturbations(self, data, auxi=None, mode=1):
		all_pers = [self.if_scale_aug, self.if_filp_aug, self.if_rotation_aug, self.if_intensity_aug, \
					self.if_noise_aug, self.if_blur_aug, self.if_mask_aug, self.if_sobel_aug, \
					self.if_mixup_aug, self.if_misalign_aug, self.if_elastic_aug, self.if_artifact_aug, \
					self.if_missing_aug, self.if_blurenhanced_aug]
		if mode == 1:
			# select used perturbations
			used_pers = []
			for k, value in enumerate(all_pers):
				if value:
					used_pers.append(k)
			# select which one perturbation to use
			if len(used_pers) == 0:
				# do nothing
				# must crop
				data = data[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				scale_size = data.shape[-1]
				rule = np.asarray([0,0,0,0], dtype=np.int32)
				rotnum = 0
				return data, scale_size, rule, rotnum
			elif len(used_pers) == 1:
				# No choise if only one perturbation can be used
				rand_per = used_pers[0]
			else:
				rand_per = random.choice(used_pers)
			# do augmentation
			# resize
			if rand_per == 0:
				data, scale_size = self.per_rescale(data)
			else:
				data = data[:, self.sub_padding[-1]:-self.sub_padding[-1], self.sub_padding[-1]:-self.sub_padding[-1]]
				scale_size = data.shape[-1]
			# flip
			if rand_per == 1:
				data, rule = self.per_flip(data)
			else:
				rule = np.asarray([0,0,0,0], dtype=np.int32)
			# rotation
			if rand_per == 2:
				rotnum = random.randint(0, 3)
				data = np.rot90(data, k=rotnum, axes=(1,2))
			else:
				rotnum = 0
			# intensity
			if rand_per == 3:
				data = self.per_intensity(data)
			# noise
			if rand_per == 4:
				data = self.per_gaussnoise(data)
			# blur
			if rand_per == 5:
				data = self.per_gaussblur(data)
			# mask or cutout
			if rand_per == 6:
				data = self.per_cutout(data)
			# sobel
			if rand_per == 7:
				data = self.per_sobel(data)
			# mixup
			if rand_per == 8 and auxi is not None:
				data = self.per_mixup(data, auxi)
			# misalign
			if rand_per == 9:
				data = self.per_misalign(data)
			# elastic
			if rand_per == 10:
				data = self.per_elastic(data)
			# artifact
			if rand_per == 11:
				data = self.per_artifact(data)
			# missing section
			if rand_per == 12:
				data = self.per_missing(data)
			# blur enhanced
			if rand_per == 13:
				data = self.per_blurenhanced(data)
		else:
			raise NotImplementedError
		return data, scale_size, rule, rotnum

	def __len__(self):
		return int(sys.maxsize)


class Provider(object):
	def __init__(self, stage, cfg):
			#patch_size, batch_size, num_workers, is_cuda=True):
		self.stage = stage
		if self.stage == 'train':
			self.data = Train(cfg)
			self.batch_size = cfg.TRAIN.batch_size
			self.num_workers = cfg.TRAIN.num_workers
		elif self.stage == 'valid':
			# return valid(folder_name, kwargs['data_list'])
			pass
		else:
			raise AttributeError('Stage must be train/valid')
		self.is_cuda = cfg.TRAIN.if_cuda
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1
	
	def __len__(self):
		return self.data.num_per_epoch
	
	def build(self):
		if self.stage == 'train':
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
		else:
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))
	
	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = next(self.data_iter)
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
				batch[3] = batch[3].cuda()
				batch[4] = batch[4].cuda()
			return batch
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = next(self.data_iter)
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
				batch[3] = batch[3].cuda()
				batch[4] = batch[4].cuda()
			return batch

def show(img3d):
	# only used for image with shape [18, 160, 160]
	num = img3d.shape[0]
	column = 5
	row = math.ceil(num / float(column))
	size = img3d.shape[1]
	img_all = np.zeros((size*row, size*column), dtype=np.uint8)
	for i in range(row):
		for j in range(column):
			index = i*column + j
			if index >= num:
				img = np.zeros_like(img3d[0], dtype=np.uint8)
			else:
				img = (img3d[index] * 255).astype(np.uint8)
			img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
	return img_all


if __name__ == '__main__':
	import yaml
	from attrdict import AttrDict
	from utils.show import show_one
	""""""
	seed = 555
	np.random.seed(seed)
	random.seed(seed)
	cfg_file = 'seg_snemi3d_d5_1024_u200.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	out_path = os.path.join('./', 'data_temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	t = time.time()
	for i in range(0, 50):
		t1 = time.time()
		per_data, gt_data, det_size, _, _ = iter(data).__next__()
		# print('single cost time: ', time.time()-t1)
		print('det_size=%d' % det_size[0])
		per_data = np.squeeze(per_data)
		gt_data = np.squeeze(gt_data)
		if cfg.MODEL.model_type == 'mala':
			per_data = per_data[14:-14,106:-106,106:-106]
			gt_data = gt_data[14:-14,106:-106,106:-106]

		img_data = show_one(per_data)
		img_affs = show_one(gt_data)
		im_cat = np.concatenate([img_data, img_affs], axis=1)
		Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
	print(time.time() - t)