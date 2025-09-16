import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset
# option file should be modified according to your expriment
from options import Option
import torchvision.transforms as transforms
from dataloader import DataLoader
from trainer_direct import Trainer
import shutil
import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d
import pickle
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from collections import OrderedDict
# from regularizer import get_reg_criterions
from models import ResNet18, ResNet50

# Classification task datasets from MedMNIST2D
CLASSIFICATION_DATASETS = {
    'pathmnist': 9,      # Colon Pathology - Multi-Class (9)
    'dermamnist': 7,     # Dermatoscope - Multi-Class (7)
    'octmnist': 4,       # Retinal OCT - Multi-Class (4)
    'pneumoniamnist': 2, # Chest X-Ray - Binary-Class (2)
    'retinamnist': 5,    # Fundus Camera - Ordinal Regression (5) - treated as classification
    'breastmnist': 2,    # Breast Ultrasound - Binary-Class (2)
    'bloodmnist': 8,     # Blood Cell Microscope - Multi-Class (8)
    'tissuemnist': 8,    # Kidney Cortex Microscope - Multi-Class (8)
    'organamnist': 11,   # Abdominal CT - Multi-Class (11)
    'organcmnist': 11,   # Abdominal CT - Multi-Class (11)
    'organsmnist': 11,   # Abdominal CT - Multi-Class (11)
}

medmnist_dataset = list(CLASSIFICATION_DATASETS.keys())


class Generator_32(nn.Module):
	def __init__(self, options=None, conf_path=None):
		super(Generator_32, self).__init__()
		self.settings = options or Option(conf_path)
		self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels):
		gen_input = torch.mul(self.label_emb(labels), z)
		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0(out)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2(img)
		return img

class Generator_224(nn.Module):
	def __init__(self, options=None, conf_path=None):
		self.settings = options or Option(conf_path)

		super(Generator_224, self).__init__()

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(self.settings.nClasses, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(self.settings.nClasses, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(self.settings.nClasses, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img


def create_generator(options=None, conf_path=None):
	if options is not None:
		settings = options
	elif conf_path is not None:
		settings = Option(conf_path)
	else:
		raise ValueError("options or conf_path must be provided")

	if settings.img_size == 32:
		return Generator_32(options=options, conf_path=conf_path)
	elif settings.img_size == 28:
		# For 28x28 models, use Generator_32 as they have similar architecture
		return Generator_32(options=options, conf_path=conf_path)
	elif settings.img_size == 224:
		return Generator_224(options=options, conf_path=conf_path)
	# elif image_size == 64:
	#     return Generator_64(options=options, conf_path=conf_path) # 향후 확장 가능
	else:
		raise ValueError(f"Unsupported image size: {settings.img_size}")


class direct_dataset(Dataset):
	def __init__(self, settings, logger, dataset):
		self.settings = settings
		self.logger = logger
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])

		# 이미지 크기 기반으로 transform 설정
		if settings.img_size in [28, 32]:
			self.train_transform = transforms.Compose([
				transforms.RandomResizedCrop(size=settings.img_size, scale=(0.5, 1.0)),
				transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
				transforms.RandomHorizontalFlip(),
			])
		else:
			self.train_transform = transforms.Compose([
				transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
				transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
				transforms.RandomHorizontalFlip(),
			])

		self.tmp_data = None
		self.tmp_label = None
		for i in range(1,5):
			# data!
			path = self.settings.generateDataPath +str(i)+".pickle"
			self.logger.info(path)
			with open(path, "rb") as fp:  # Pickling
				gaussian_data = pickle.load(fp)
			
			# import IPython
			# IPython.embed()
			if self.tmp_data is None:
				self.tmp_data = np.concatenate(gaussian_data, axis=0)
			else:
				self.tmp_data = np.concatenate((self.tmp_data, np.concatenate(gaussian_data, axis=0)))

			# label!info
			path = self.settings.generateLabelPath + str(i) + ".pickle"
			self.logger.info(path)
			with open(path, "rb") as fp:  # Pickling
				labels_list = pickle.load(fp)
			if self.tmp_label is None:
				self.tmp_label = np.concatenate(labels_list, axis=0)
			else:
				self.tmp_label = np.concatenate((self.tmp_label, np.concatenate(labels_list, axis=0)))

		assert len(self.tmp_label) == len(self.tmp_data)
		# print(self.tmp_data.shape, self.tmp_label.shape)
		# print('direct datset image number', len(self.tmp_label))


	def __getitem__(self, index):
		img = self.tmp_data[index]
		label = self.tmp_label[index]
		img = self.train_transform(torch.from_numpy(img))
		return img, label

	def __len__(self):
		return len(self.tmp_label)


def convert_state_dict(pretrained_state_dict, new_model, model_type='standard'):
    """
    Converts a pretrained ResNet-18 state_dict to match the key format of the new model.
    
    Args:
        pretrained_state_dict (OrderedDict): The state_dict object from the pretrained model.
        new_model (torch.nn.Module): An instance of the new model architecture.
        model_type (str): Type of model ('28x28' or 'standard').

    Returns:
        OrderedDict: The converted state_dict.
    """
    if model_type == '28x28':
        # For 28x28 models, need to convert from checkpoint naming to our model naming
        new_state_dict = OrderedDict()
        key_map = {}
        
        # Map shortcut -> downsample and linear -> fc
        for key, value in pretrained_state_dict.items():
            new_key = key
            if 'shortcut' in key:
                new_key = key.replace('shortcut', 'downsample')
            elif 'linear' in key:
                new_key = key.replace('linear', 'fc')
            key_map[key] = new_key
        
        # Apply the mapping
        for old_key, new_key in key_map.items():
            new_state_dict[new_key] = pretrained_state_dict[old_key]
        
        return new_state_dict
    
    # For standard 224x224 models, use the existing conversion logic
    # For debugging: Check if the problematic key exists in the source state_dict
    if 'bn1.num_batches_tracked' not in pretrained_state_dict:
        print("Warning: 'bn1.num_batches_tracked' not found in the source checkpoint!")

    new_state_dict = OrderedDict()
    key_map = {}

    # 1. Initial block (conv1 and bn1)
    key_map.update({
        'conv1.weight': 'features.init_block.conv.conv.weight',
        'bn1.weight': 'features.init_block.conv.bn.weight',
        'bn1.bias': 'features.init_block.conv.bn.bias',
        'bn1.running_mean': 'features.init_block.conv.bn.running_mean',
        'bn1.running_var': 'features.init_block.conv.bn.running_var',
        'bn1.num_batches_tracked': 'features.init_block.conv.bn.num_batches_tracked'  # The missing key
    })

    # 2. ResNet stages (layer1 to layer4)
    for i in range(1, 5):  # Stages 1-4
        for j in range(2):  # Units 1-2 (for ResNet-18)
            # Body convolutions and their batchnorms
            for conv_idx in [1, 2]:
                old_prefix = f'layer{i}.{j}.conv{conv_idx}'
                new_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_prefix}.weight'] = f'{new_prefix}.conv.weight'

                old_bn_prefix = f'layer{i}.{j}.bn{conv_idx}'
                new_bn_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_bn_prefix}.weight'] = f'{new_bn_prefix}.bn.weight'
                key_map[f'{old_bn_prefix}.bias'] = f'{new_bn_prefix}.bn.bias'
                key_map[f'{old_bn_prefix}.running_mean'] = f'{new_bn_prefix}.bn.running_mean'
                key_map[f'{old_bn_prefix}.running_var'] = f'{new_bn_prefix}.bn.running_var'
                key_map[f'{old_bn_prefix}.num_batches_tracked'] = f'{new_bn_prefix}.bn.num_batches_tracked'

            # Downsample (identity) convolution for stages 2, 3, 4
            if i > 1 and j == 0:
                old_ds_prefix = f'layer{i}.{j}.downsample'
                new_ds_prefix = f'features.stage{i}.unit{j+1}.identity_conv'
                key_map[f'{old_ds_prefix}.0.weight'] = f'{new_ds_prefix}.conv.weight'
                key_map[f'{old_ds_prefix}.1.weight'] = f'{new_ds_prefix}.bn.weight'
                key_map[f'{old_ds_prefix}.1.bias'] = f'{new_ds_prefix}.bn.bias'
                key_map[f'{old_ds_prefix}.1.running_mean'] = f'{new_ds_prefix}.bn.running_mean'
                key_map[f'{old_ds_prefix}.1.running_var'] = f'{new_ds_prefix}.bn.running_var'
                key_map[f'{old_ds_prefix}.1.num_batches_tracked'] = f'{new_ds_prefix}.bn.num_batches_tracked'

    # 3. Final fully-connected layer
    key_map.update({
        'fc.weight': 'output.weight',
        'fc.bias': 'output.bias'
    })

    # Populate the new_state_dict using the generated map
    for old_key, new_key in key_map.items():
        if old_key in pretrained_state_dict:
            new_state_dict[new_key] = pretrained_state_dict[old_key]

    return new_state_dict


class ExperimentDesign:
	def __init__(self, options=None, args=None, logger=None):
		self.settings = options
		self.args = args
		self.logger = logger

		self.train_loader = None
		self.test_loader = None
		self.model = None
		self.model_teacher = None
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0

		self.prepare()
	
	def set_logger(self):
		# logger = logging.getLogger('baseline')
		if dist.get_rank()==0:
			file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
			file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
			file_handler.setFormatter(file_formatter)
			self.logger.addHandler(file_handler)
		self.logger.setLevel(logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN)
		return self.logger

	def prepare(self):
		torch.cuda.set_device(self.args.local_rank)
		dist.init_process_group(backend='nccl')
		if dist.get_rank() == 0:
			self.settings.set_save_path()
			shutil.copyfile(self.args.conf_path, os.path.join(self.settings.save_path, os.path.basename(self.args.conf_path)))
			shutil.copyfile('./main_direct.py', os.path.join(self.settings.save_path, 'main_direct.py'))
			shutil.copyfile('./trainer_direct.py', os.path.join(self.settings.save_path, 'trainer_direct.py'))
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)
		self._set_gpu()
		self._set_dataloader()
		self._set_model()
		self._replace()
		self.logger.info(self.model_teacher)
		self.logger.info(self.model)
		self._set_trainer()
	
	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		data_loader = DataLoader(dataset=self.settings.dataset,
		                         batch_size=self.settings.batchSize,
		                         data_path=self.settings.dataPath,
		                         n_threads=self.settings.nThreads,
		                         ten_crop=self.settings.tenCrop,
		                         logger=self.logger)
		
		self.train_loader, self.test_loader = data_loader.getloader()

	def _set_model(self):

		if self.settings.dataset in medmnist_dataset:
			num_classes = self.settings.nClasses
			if self.settings.img_size == 28:
				self.model = ResNet18(in_channels=3, num_classes=num_classes, img_size=self.settings.img_size)
				self.model_teacher = ResNet18(in_channels=3, num_classes=num_classes, img_size=self.settings.img_size)
				print(f'****** 28x28 ResNet18 model created with {num_classes} classes for {self.settings.dataset} ******')

				# Load checkpoint and handle different formats
				checkpoint = torch.load(self.settings.pretrained_path, map_location='cpu')
				
				self.model.load_state_dict(checkpoint['net'])
				self.model_teacher.load_state_dict(checkpoint['net'])
				print(f'****** Pretrained model {self.settings.pretrained_path} loaded ******')
			else:
				self.model = ptcv_get_model(self.settings.model_name, pretrained=False, num_classes=num_classes)
				self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=False, num_classes=num_classes)
				print(f'****** Model created with {num_classes} classes for {self.settings.dataset} ******')

				# Load checkpoint and handle different formats
				checkpoint = torch.load(self.settings.pretrained_path, map_location='cpu')
				if isinstance(checkpoint, dict) and 'net' in checkpoint:
					converted_state_dict = convert_state_dict(checkpoint['net'], self.model)
				else:
					converted_state_dict = convert_state_dict(checkpoint, self.model)
			
				self.model.load_state_dict(converted_state_dict)
				self.model_teacher.load_state_dict(converted_state_dict)
				print(f'****** Pretrained model {self.settings.pretrained_path} loaded ******')

		else:
			self.model = ptcv_get_model(self.settings.model_name, pretrained=True)
			self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=True)

		self.generator = create_generator(options=self.settings)
		self.model_teacher.eval()
		
		self.model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model_teacher)
		self.model_teacher = DDP(self.model_teacher.to(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False)
		self.generator = DDP(self.generator.to(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False)

	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
		                           self.settings.nEpochs,
		                           self.settings.lrPolicy_S)
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}

		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}
		
		lr_master_S.set_params(params_dict=params_dict_S)
		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			generator = self.generator,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=lr_master_S,
			lr_master_G=lr_master_G,
			settings=self.settings,
			args = self.args,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			run_count=self.start_epoch)

	def quantize_model(self,model):
		"""
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""
		
		weight_bit = self.settings.qw
		act_bit = self.settings.qa
		
		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		
		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
		
		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model(m))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model(mod))
			return q_model
	
	def _replace(self):
		self.model = self.quantize_model(self.model)
		self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
		self.model = DDP(self.model.to(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False)
	
	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model

	def run(self):
		best_top1 = 100
		best_top5 = 100
		start_time = time.time()

		dataset = direct_dataset(self.settings, self.logger, self.settings.dataset)

		direct_dataload = torch.utils.data.DataLoader(dataset,
													   batch_size=min(self.settings.batchSize, len(dataset)),
													   sampler = DistributedSampler(dataset))
		try:
			for epoch in tqdm(range(self.start_epoch, self.settings.nEpochs), desc="Training Progress"):
				self.epoch = epoch
				self.start_epoch = 0

				if epoch < 4:
					self.unfreeze_model(self.model)

				train_error, train_loss, train5_error = self.trainer.train(epoch=epoch, direct_dataload=direct_dataload)

				self.freeze_model(self.model)

				# 모든 데이터셋에 대해 동일한 테스트 로직 적용
				if epoch >= 0:
					test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
				else:
					test_error = 100
					test5_error = 100

				if best_top1 >= test_error:
					best_top1 = test_error
					best_top5 = test5_error
					# self.logger.info(
					# 	'Save model! The path is ' + os.path.join(self.settings.save_path, "model.pth"))
					# if dist.get_rank() == 0:
					# 	torch.save(self.model.state_dict(), os.path.join(self.settings.save_path, "model.pth"))
				
				self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
				self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1,
				                                                                                       100 - best_top5))

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return best_top1, best_top5


def main():
	logger = logging.getLogger()
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument("--local_rank", type=int, default=None)
	args = parser.parse_args()

	# torchrun이 넘겨주는 환경변수 우선 적용
	if args.local_rank is None or args.local_rank == -1:
		import os
		args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

	option = Option(args.conf_path)
	option.manualSeed = 1

	experiment = ExperimentDesign(option, args, logger)
	experiment.run()


if __name__ == '__main__':
	main()
