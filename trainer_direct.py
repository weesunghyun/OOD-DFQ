"""
basic trainer
"""
from builtins import isinstance
from importlib.machinery import DEBUG_BYTECODE_SUFFIXES
import time
import os
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from collections import OrderedDict
from pytorchcv.models.resnet import ResUnit
# from pytorchcv.models.mobilenet import DwsConvBlock  # 이 import는 사용하지 않음
from pytorchcv.models.mobilenetv2 import LinearBottleneck
from quantization_utils.quant_modules import *
import torch.distributed as dist
try:
	from torch.utils.tensorboard import SummaryWriter
except ImportError:
	SummaryWriter = None


__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
	             train_loader, test_loader, settings, args, logger, tensorboard_logger=None,
	             opt_type="SGD", optimizer_state=None, run_count=0):
		"""
		init trainer
		"""
		
		self.settings = settings
		self.args = args
		self.model = model
		self.model_teacher = model_teacher
		self.generator = generator
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tensorboard_logger = tensorboard_logger
		# self.tensorboard_logger = SummaryWriter(self.settings.save_path) if dist.get_rank()==0 else None
		self.criterion = nn.CrossEntropyLoss().to(self.args.local_rank)
		self.bce_logits = nn.BCEWithLogitsLoss().to(self.args.local_rank)
		self.MSE_loss = nn.MSELoss().to(self.args.local_rank)
		self.KLloss = nn.KLDivLoss(reduction='batchmean').to(self.args.local_rank)
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type
		if opt_type == "SGD":
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
			)
		elif opt_type == "RMSProp":
			self.optimizer_S = torch.optim.RMSprop(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1.0,
				weight_decay=self.settings.weightDecay,
				momentum=self.settings.momentum,
				alpha=self.settings.momentum
			)
		elif opt_type == "Adam":
			self.optimizer_S = torch.optim.Adam(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1e-5,
				weight_decay=self.settings.weightDecay
			)
		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)\

		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
											betas=(self.settings.b1, self.settings.b2))

		self.logger = logger
		self.run_count = run_count
		self.scalar_info = {}
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []
		self.save_BN_mean = []
		self.save_BN_var = []
		self.activation_teacher = []
		self.activation = []
		self.handle_list = []

		self.bsdc_start_epoch = int(getattr(self.settings, 'bsdc_start_epoch', self.settings.nEpochs - 1))
		if self.bsdc_start_epoch < 0:
			self.bsdc_start_epoch = 0
		self.bsdc_num_batches = getattr(self.settings, 'bsdc_num_batches', None)
		if self.bsdc_num_batches is not None:
			self.bsdc_num_batches = int(self.bsdc_num_batches)
		if self.bsdc_num_batches is not None and self.bsdc_num_batches <= 0:
			self.bsdc_num_batches = None
		self.bsdc_correction_applied = False
		self.bn_layer_names = []
		self.teacher_bn_layers = []
		self.student_bn_layers = []
		self.teacher_bn_source_stats = []
		self.bsdc_delta_means = []
		self.bsdc_delta_vars = []
		self.bsdc_teacher_ood_stats = []
		self.bsdc_student_ood_stats = []
		self._init_bn_tracking()

	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_S = self.lr_master_S.get_lr(epoch)
		lr_G = self.lr_master_G.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_S.param_groups:
			param_group['lr'] = lr_S
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr_G

	def _init_bn_tracking(self):
		"""Prepare BatchNorm bookkeeping for BSDC."""
		if not hasattr(self.model_teacher, 'module') or not hasattr(self.model, 'module'):
			return

		teacher_bn_dict = OrderedDict(
			(name, module)
			for name, module in self.model_teacher.module.named_modules()
			if isinstance(module, nn.modules.batchnorm._BatchNorm)
		)
		student_bn_dict = OrderedDict(
			(name, module)
			for name, module in self.model.module.named_modules()
			if isinstance(module, nn.modules.batchnorm._BatchNorm)
		)

		common_names = [name for name in teacher_bn_dict.keys() if name in student_bn_dict]
		for name in common_names:
			teacher_bn = teacher_bn_dict[name]
			student_bn = student_bn_dict[name]
			self.bn_layer_names.append(name)
			self.teacher_bn_layers.append(teacher_bn)
			self.student_bn_layers.append(student_bn)
			self.teacher_bn_source_stats.append({
				'mean': teacher_bn.running_mean.detach().clone(),
				'var': teacher_bn.running_var.detach().clone()
			})

		if self.logger is not None and len(self.teacher_bn_layers) == 0:
			self.logger.warning('BSDC: No matching BatchNorm layers were found; BSDC will be skipped.')
		elif self.logger is not None and len(self.teacher_bn_layers) != len(self.student_bn_layers):
			self.logger.warning(
				'BSDC: Mismatched number of BatchNorm layers between teacher (%d) and student (%d).',
				len(self.teacher_bn_layers), len(self.student_bn_layers)
			)

		self.bsdc_delta_means = [torch.zeros_like(stats['mean']) for stats in self.teacher_bn_source_stats]
		self.bsdc_delta_vars = [torch.zeros_like(stats['var']) for stats in self.teacher_bn_source_stats]
		self.bsdc_teacher_ood_stats = [None for _ in self.teacher_bn_layers]
		self.bsdc_student_ood_stats = [None for _ in self.student_bn_layers]

	def _create_bn_stat_hook(self, mean_store, var_store, count_store, index):
		def hook(module, inputs, outputs):
			if not inputs:
				return
			input_tensor = inputs[0]
			if input_tensor.dim() <= 1:
				return
			dims = [0]
			if input_tensor.dim() > 2:
				dims.extend(range(2, input_tensor.dim()))
			mean = input_tensor.mean(dim=dims)
			var = input_tensor.var(dim=dims, unbiased=False)
			if dist.is_available() and dist.is_initialized():
				mean_clone = mean.clone()
				var_clone = var.clone()
				dist.all_reduce(mean_clone, op=dist.ReduceOp.SUM)
				dist.all_reduce(var_clone, op=dist.ReduceOp.SUM)
				world_size = dist.get_world_size()
				mean = mean_clone / world_size
				var = var_clone / world_size
			mean_store[index] += mean.detach()
			var_store[index] += var.detach()
			count_store[index] += 1
		return hook

	def apply_bsdc_correction(self, data_loader, epoch):
		"""Apply BN Stat Delta Correction using the provided OOD data loader."""
		if self.bsdc_correction_applied:
			return
		if len(self.teacher_bn_layers) == 0 or len(self.student_bn_layers) == 0:
			self.bsdc_correction_applied = True
			return
		if data_loader is None:
			if self.logger is not None:
				self.logger.warning('BSDC: OOD dataloader is None. Skipping BN correction.')
			self.bsdc_correction_applied = True
			return

		teacher_means_accum = [torch.zeros_like(mod.running_mean) for mod in self.teacher_bn_layers]
		teacher_vars_accum = [torch.zeros_like(mod.running_var) for mod in self.teacher_bn_layers]
		student_means_accum = [torch.zeros_like(mod.running_mean) for mod in self.student_bn_layers]
		student_vars_accum = [torch.zeros_like(mod.running_var) for mod in self.student_bn_layers]
		teacher_counts = [0 for _ in self.teacher_bn_layers]
		student_counts = [0 for _ in self.student_bn_layers]

		handles = []
		teacher_prev_mode = self.model_teacher.training
		student_prev_mode = self.model.training

		self.model_teacher.train()
		self.model.train()

		if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
			data_loader.sampler.set_epoch(epoch + self.settings.nEpochs)

		try:
			for idx, module in enumerate(self.teacher_bn_layers):
				handles.append(module.register_forward_hook(
					self._create_bn_stat_hook(teacher_means_accum, teacher_vars_accum, teacher_counts, idx)))
			for idx, module in enumerate(self.student_bn_layers):
				handles.append(module.register_forward_hook(
					self._create_bn_stat_hook(student_means_accum, student_vars_accum, student_counts, idx)))

			batch_limit = self.bsdc_num_batches
			with torch.no_grad():
				for batch_idx, (images, _) in enumerate(data_loader):
					images = images.to(self.args.local_rank)
					_ = self.model_teacher(images)
					_ = self.model(images)
					if batch_limit is not None and (batch_idx + 1) >= batch_limit:
						break
		finally:
			for handle in handles:
				handle.remove()
			if not teacher_prev_mode:
				self.model_teacher.eval()
			if not student_prev_mode:
				self.model.eval()

		teacher_ood_means = []
		teacher_ood_vars = []
		for idx, count in enumerate(teacher_counts):
			if count > 0:
				teacher_ood_means.append(teacher_means_accum[idx] / count)
				teacher_ood_vars.append(teacher_vars_accum[idx] / count)
			else:
				teacher_ood_means.append(self.teacher_bn_layers[idx].running_mean.detach().clone())
				teacher_ood_vars.append(self.teacher_bn_layers[idx].running_var.detach().clone())

		student_ood_means = []
		student_ood_vars = []
		for idx, count in enumerate(student_counts):
			if count > 0:
				student_ood_means.append(student_means_accum[idx] / count)
				student_ood_vars.append(student_vars_accum[idx] / count)
			else:
				student_ood_means.append(self.student_bn_layers[idx].running_mean.detach().clone())
				student_ood_vars.append(self.student_bn_layers[idx].running_var.detach().clone())

                for idx in range(len(self.student_bn_layers)):
                        delta_mean = student_ood_means[idx] - teacher_ood_means[idx]
                        delta_var = student_ood_vars[idx] - teacher_ood_vars[idx]
			self.bsdc_delta_means[idx] = delta_mean.detach()
			self.bsdc_delta_vars[idx] = delta_var.detach()
			self.bsdc_teacher_ood_stats[idx] = {
				'mean': teacher_ood_means[idx].detach(),
				'var': teacher_ood_vars[idx].detach()
			}
			self.bsdc_student_ood_stats[idx] = {
				'mean': student_ood_means[idx].detach(),
				'var': student_ood_vars[idx].detach()
			}

			corrected_mean = self.teacher_bn_source_stats[idx]['mean'] + delta_mean
			corrected_var = self.teacher_bn_source_stats[idx]['var'] + delta_var
			corrected_var = torch.clamp(corrected_var, min=1e-6)
			self.student_bn_layers[idx].running_mean.data.copy_(corrected_mean)
			self.student_bn_layers[idx].running_var.data.copy_(corrected_var)

		for module, src_stats in zip(self.teacher_bn_layers, self.teacher_bn_source_stats):
			module.running_mean.data.copy_(src_stats['mean'])
			module.running_var.data.copy_(src_stats['var'])

		if self.logger is not None:
			processed_batches = max(teacher_counts) if len(teacher_counts) > 0 else 0
			self.logger.info(
				'BSDC: Applied BN statistic delta correction using %d batches across %d layers.',
				processed_batches,
				len(self.student_bn_layers)
			)

		self.bsdc_correction_applied = True
	def loss_fn_kd(self, output, labels, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""	
		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, dim=1)
		b = F.softmax(teacher_outputs / T, dim=1)
		c = (alpha * T * T)
		d = self.criterion(output, labels)
		KD_loss = self.KLloss(a, b) * c
		return KD_loss, d

	def loss_fa(self):
		fa = torch.zeros(1).to(self.args.local_rank)
		for l in range(len(self.activation)):
			fa += (self.activation[l] - self.activation_teacher[l]).pow(2).mean()
		fa = self.settings.lam * fa
		return fa
	
	def forward(self, images, teacher_outputs, labels=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize
		output = self.model(images)
		loss_KL, loss_CE = self.loss_fn_kd(output, labels, teacher_outputs)
		loss_FA = self.loss_fa()
		return output, loss_KL, loss_FA, loss_CE
	
	def backward_G(self, loss_G):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		loss_G.backward()
		self.optimizer_G.step()

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss_S.backward()
		self.optimizer_S.step()

	def backward(self, loss):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		self.optimizer_S.zero_grad()
		loss.backward()
		self.optimizer_G.step()
		self.optimizer_S.step()

	def reduce_minmax(self):
		for m in self.model.module.modules():
			if isinstance(m, QuantAct):
				dist.all_reduce(m.x_min, op=dist.ReduceOp.SUM)
				dist.all_reduce(m.x_max, op=dist.ReduceOp.SUM)
				m.x_min = m.x_min / dist.get_world_size()
				m.x_max = m.x_max / dist.get_world_size()

	def spatial_attention(self, x):
		return F.normalize(x.pow(2).mean([1]).view(x.size(0), -1))

	def channel_attention(self, x):
		return F.normalize(x.pow(2).mean([2,3]).view(x.size(0), -1))

	def hook_activation_teacher(self, module, input, output):
		self.activation_teacher.append(self.channel_attention(output.clone()))

	def hook_activation(self, module, input, output):
		self.activation.append(self.channel_attention(output.clone()))

	def hook_fn_forward(self,module, input, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		# use biased var in train
		var = input.var([0, 2, 3], unbiased=False)

		self.mean_list.append(mean)
		self.var_list.append(var)
		self.teacher_running_mean.append(module.running_mean)
		self.teacher_running_var.append(module.running_var)
	
	def train(self, epoch, direct_dataload=None):
		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()

		iters = 200
		self.update_lr(epoch)

		self.model.eval()
		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time
		
		if epoch==0:
			#register BN hook
			for m in self.model_teacher.module.modules():
				if isinstance(m, nn.SyncBatchNorm):
					handle = m.register_forward_hook(self.hook_fn_forward)
					self.handle_list.append(handle)

		if epoch == 4:
			#remove BN hook
			for handle in self.handle_list:
				handle.remove()
			self.reduce_minmax()

			for m in self.model_teacher.module.modules():
				if isinstance(m, ResUnit):
					m.body.register_forward_hook(self.hook_activation_teacher)
				elif isinstance(m, LinearBottleneck):
					m.conv3.register_forward_hook(self.hook_activation_teacher)
			for m in self.model.module.modules():
				if isinstance(m, ResUnit):
					m.body.register_forward_hook(self.hook_activation)
				elif isinstance(m, LinearBottleneck):
					m.conv3.register_forward_hook(self.hook_activation)

			self.generator = self.generator.cpu()
			self.optimizer_G.zero_grad()
			# del self.optimizer_G
			# torch.cuda.empty_cache()

		if direct_dataload is not None:
			direct_dataload.sampler.set_epoch(epoch)
			iterator = iter(direct_dataload)

		for i in range(iters):

			start_time = time.time()
			data_time = start_time - end_time

			# torch.cuda.empty_cache()

			if epoch < 4:
				z = Variable(torch.randn(16, self.settings.latent_dim)).to(self.args.local_rank)
				labels = Variable(torch.randint(0, self.settings.nClasses, (16,))).to(self.args.local_rank)
				z = z.contiguous()
				labels = labels.contiguous()
				images = self.generator(z, labels)

				self.mean_list.clear()
				self.var_list.clear()
				self.teacher_running_mean.clear()
				self.teacher_running_var.clear()
				output_teacher_batch = self.model_teacher(images)

				# One hot loss
				loss_one_hot = self.criterion(output_teacher_batch, labels)
				# BN statistic loss
				BNS_loss = torch.zeros(1).to(self.args.local_rank)
				for num in range(len(self.mean_list)):
					BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
						self.var_list[num], self.teacher_running_var[num])

				num_bn = len(self.mean_list)
				if num_bn == 0:
					print("Warning: no BatchNorm hooks triggered; skipping BN losses.")
					loss_G = loss_one_hot
				else:
					BNS_loss = BNS_loss / num_bn
					# loss of Generator
					loss_G = loss_one_hot + 0.1 * BNS_loss
				self.backward_G(loss_G)
				output = self.model(images.detach())
				loss_S = torch.zeros(1).to(self.args.local_rank)
			else:

				try:
					images, labels = next(iterator)
				except:
					# self.logger.info('re-iterator of direct_dataload')
					iterator = iter(direct_dataload)
					images, labels = next(iterator)
				images, labels = images.to(self.args.local_rank), labels.to(self.args.local_rank)

				self.activation_teacher.clear()
				self.activation.clear()

				images.requires_grad = True
				output_teacher_batch = self.model_teacher(images)
				output, loss_KL, loss_FA, loss_CE = self.forward(images, output_teacher_batch, labels)
				loss_S = loss_KL + loss_FA

				perturbation = torch.sgn(torch.autograd.grad(loss_S, images, retain_graph=True)[0])
				self.activation_teacher.clear()
				self.activation.clear()
				with torch.no_grad():
					images_perturbed = images + self.settings.eps * perturbation
					output_teacher_batch_perturbed = self.model_teacher(images_perturbed.detach())
				output_perturbed, loss_KL_perturbed, loss_FA_perturbed, loss_CE_perturbed = self.forward(images_perturbed.detach(), output_teacher_batch_perturbed.detach(), labels)
				loss_S_perturbed = loss_KL_perturbed + loss_FA_perturbed
				loss_total = 1 * loss_S + 1 * loss_S_perturbed

				self.backward_S(loss_total)

			single_error, single_loss, single5_error = utils.compute_singlecrop(
				outputs=output, labels=labels,
				loss=loss_S, top5_flag=True, mean_flag=True)
			
			top1_error.update(single_error, images.size(0))
			top1_loss.update(single_loss, images.size(0))
			top5_error.update(single5_error, images.size(0))
			
			end_time = time.time()
			
			gt = labels.data.cpu().numpy()
			d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)
			fp_acc.update(d_acc)
		
		if epoch < 4:
			self.logger.info(
				"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, 100 * fp_acc.avg, loss_G.item(),
				   loss_one_hot.item(), BNS_loss.item())
			)
		else:
			self.logger.info(
				"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [loss KL: %f] [loss FA: %f] [loss perturbed KL: %f] [loss perturbed FA: %f] "
				% (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_KL.item(), loss_FA.item(), loss_KL_perturbed.item(), loss_FA_perturbed.item())
			)

		# self.scalar_info['accuracy every epoch'] = 100 * d_acc
		# self.scalar_info['training_top1error'] = top1_error.avg
		# self.scalar_info['training_top5error'] = top5_error.avg
		# self.scalar_info['training_loss'] = top1_loss.avg
		
		# if self.tensorboard_logger is not None:
		# 	for tag, value in list(self.scalar_info.items()):
		# 		self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
		# 	self.scalar_info = {}

		if (not self.bsdc_correction_applied) and (epoch >= self.bsdc_start_epoch):
			self.apply_bsdc_correction(direct_dataload, epoch)

		return top1_error.avg, top1_loss.avg, top5_error.avg


	def test(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time
		# g=[]
		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.to(self.args.local_rank)
				images = images.to(self.args.local_rank)
				# print(self.model)
				# print(images.shape)
				# print(labels.shape)
				# breakpoint()
				output = self.model(images)
				loss = torch.ones(1)
				self.mean_list.clear()
				self.var_list.clear()

				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		self.logger.info(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		
		# self.scalar_info['testing_top1error'] = top1_error.avg
		# self.scalar_info['testing_top5error'] = top5_error.avg
		# self.scalar_info['testing_loss'] = top1_loss.avg
		# if self.tensorboard_logger is not None:
		# 	for tag, value in self.scalar_info.items():
		# 		self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
		# 	self.scalar_info = {}
		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg


	def test_teacher(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.to(self.args.local_rank)

				if self.settings.tenCrop:
					image_size = images.size()
					images = images.view(
						image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.to(self.args.local_rank)
						img_var = Variable(img, volatile=True)
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = torch.cat((output, temp_output.data))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.to(self.args.local_rank)
					self.activation_teacher.clear()
					output = self.model_teacher(images)

					loss = torch.ones(1)
					self.mean_list.clear()
					self.var_list.clear()

					single_error, single_loss, single5_error = utils.compute_singlecrop(
						outputs=output, loss=loss,
						labels=labels, top5_flag=True, mean_flag=True)
				#
				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))

				end_time = time.time()
				iter_time = end_time - start_time

		print(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		self.run_count += 1

		return top1_error.avg, top1_loss.avg, top5_error.avg
