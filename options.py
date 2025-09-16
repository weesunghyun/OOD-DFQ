import os
import shutil
import uuid
from pyhocon import ConfigFactory

from utils.opt_static import NetOption


class Option(NetOption):
	def __init__(self, conf_path):
		super(Option, self).__init__()
		self.conf = ConfigFactory.parse_file(conf_path)
		#  ------------ General options ----------------------------------------
		self.model_name = self.conf['model_name']
		self.generateDataPath = self.conf['generateDataPath']
		self.generateLabelPath = self.conf['generateLabelPath']
		self.dataPath = self.conf['dataPath']  # path for loading data set
		self.dataset = self.conf['dataset']  # options: imagenet | cifar100
		
		# ------------- Data options -------------------------------------------
		self.nThreads = self.conf['nThreads']  # number of data loader threads
		
		# ---------- Optimization options --------------------------------------
		self.nEpochs = self.conf['nEpochs']  # number of total epochs to train
		self.batchSize = self.conf['batchSize']  # mini-batch size
		self.momentum = self.conf['momentum']  # momentum
		self.weightDecay = float(self.conf['weightDecay'])  # weight decay
		self.opt_type = self.conf['opt_type']

		self.lr_S = self.conf['lr_S']  # initial learning rate
		self.lrPolicy_S = self.conf['lrPolicy_S']  # options: multi_step | linear | exp | const | step
		self.step_S = self.conf['step_S']  # step for linear or exp learning rate policy
		self.decayRate_S = self.conf['decayRate_S']  # lr decay rate
		
		# ---------- Model options ---------------------------------------------
		self.nClasses = self.conf['nClasses']  # number of classes in the dataset
		if self.conf['img_size'] == 28:
			self.pretrained_path = f'./checkpoints/{self.model_name}_{self.dataset}_28.pth'
		else:
			self.pretrained_path = f'./checkpoints/{self.model_name}_{self.dataset}.pth'
		
		# ---------- Quantization options ---------------------------------------------
		self.qw = self.conf['qw']
		self.qa = self.conf['qa']
		
		# ----------KD options ---------------------------------------------
		self.temperature = self.conf['temperature']
		self.alpha = self.conf['alpha']
		
		# ----------Generator options ---------------------------------------------
		self.latent_dim = self.conf['latent_dim']
		self.img_size = self.conf['img_size']
		self.channels = self.conf['channels']

		self.lr_G = self.conf['lr_G']
		self.lrPolicy_G = self.conf['lrPolicy_G']  # options: multi_step | linear | exp | const | step
		self.step_G = self.conf['step_G']  # step for linear or exp learning rate policy
		self.decayRate_G = self.conf['decayRate_G']  # lr decay rate

		self.b1 = self.conf['b1']
		self.b2 = self.conf['b2']

		# ----------- parameter --------------------------------------
		self.lam = 1000
		self.eps = 0.01

		
	def set_save_path(self):
		path='HAST_log'
		if not os.path.isdir(path):
			os.mkdir(path)
		path = os.path.join(path, self.model_name+"_"+self.dataset)
		if not os.path.isdir(path):
			os.mkdir(path)
		pathname = 'W' + str(self.qw) + 'A' + str(self.qa)
		num = int(uuid.uuid4().hex[0:4], 16)
		pathname += '_' + str(num)
		path = os.path.join(path, pathname)
		if not os.path.isdir(path):
			os.mkdir(path)
		self.save_path = path

		# self.save_path = self.save_path + "log_{}_bs{:d}_lr{:.4f}_qw{:d}_qa{:d}_epoch{}_{}/".format(
		# 	self.dataset, self.batchSize, self.lr, self.opt_type, self.qw, self.qa,
		# 	self.nEpochs, self.experimentID)
		
		# if os.path.exists(self.save_path):
		# 	print("{} file exist!".format(self.save_path))
		# 	action = input("Select Action: d (delete) / q (quit):").lower().strip()
		# 	act = action
		# 	if act == 'd':
		# 		import stat
		# 		try:
		# 			shutil.rmtree(self.save_path)
		# 		except PermissionError as e:
		# 			err_file_path = str(e).split("\'", 2)[1]
		# 			if os.path.exists(err_file_path):
		# 				os.chmod(err_file_path, stat.S_IWUSR)
		# 	else:
		# 		raise OSError("Directory {} exits!".format(self.save_path))
		
		# if not os.path.exists(self.save_path):
		# 	os.makedirs(self.save_path)
	
	def paramscheck(self, logger):
		logger.info("|===>The used PyTorch version is {}".format(
				self.torch_version))
		
		# 클래스 수를 설정 파일에서 읽어오거나 기본값 사용
		if hasattr(self, 'nClasses'):
			# 설정 파일에 nClasses가 이미 정의되어 있음
			pass
		else:
			# 기본값 설정 (하위 호환성을 위해)
			if self.dataset in ["cifar10", "mnist"]:
				self.nClasses = 10
			elif self.dataset == "cifar100":
				self.nClasses = 100
			elif self.dataset == "imagenet" or "thi_imgnet":
				self.nClasses = 1000
			elif self.dataset == "imagenet100":
				self.nClasses = 100
			elif self.dataset == 'dermamnist':
				self.nClasses = 7
			elif self.dataset == 'pathmnist':
				self.nClasses = 9
			elif self.dataset == 'octmnist':
				self.nClasses = 4
			elif self.dataset == 'pneumoniamnist':
				self.nClasses = 2
			elif self.dataset == 'breastmnist':
				self.nClasses = 2
			elif self.dataset == 'bloodmnist':
				self.nClasses = 8
			elif self.dataset == 'tissuemnist':
				self.nClasses = 8
			elif self.dataset == 'organamnist':
				self.nClasses = 11
			elif self.dataset == 'organcmnist':
				self.nClasses = 11
			elif self.dataset == 'organsmnist':
				self.nClasses = 11
			else:
				self.nClasses = 1000