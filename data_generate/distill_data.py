import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import random
import pickle
import sys

from torchvision import transforms
from torchvision import datasets

def check_path(model_path):
    """
    Check if the directory exists, if not create it.
    Args:
        model_path: path to the model
    """
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


class DistillData(object):
    def __init__(self):
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def getDistilData_hardsample(self,
                                model_name="resnet18",
                                teacher_model=None,
                                num_data=1280,
                                batch_size=256,
                                num_batch=1,
                                group=1,
                                augMargin=0.4,
                                beta=1.0,
                                gamma=0,
                                save_path_head="",
                                init_data_path=None
                                ):

        data_path = os.path.join(save_path_head, model_name+"_refined_gaussian_hardsample_" \
                    + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")
        label_path = os.path.join(save_path_head, model_name+"_labels_hardsample_" \
                    + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")

        print(data_path, label_path)

        check_path(data_path)
        check_path(label_path)

        # Prepare dataset for initialization if provided
        init_dataset = None
        if init_data_path is not None:
            if hasattr(teacher_model, 'img_size') and teacher_model.img_size == 32:
                init_transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor()
                ])
            else:
                init_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
            init_dataset = datasets.ImageFolder(init_data_path, transform=init_transform)
            init_len = len(init_dataset)
            print(f"Init dataset loaded from {init_data_path}, {init_len} images")

        # 이미지 크기 기반으로 shape 결정
        if hasattr(teacher_model, 'img_size'):
            if teacher_model.img_size == 32:
                shape = (batch_size, 3, 32, 32)
            elif teacher_model.img_size == 28:
                shape = (batch_size, 3, 28, 28)
            else:
                # 기본적으로 224 크기로 처리
                shape = (batch_size, 3, 224, 224)
        else:
            # 기본적으로 224 크기로 처리
            shape = (batch_size, 3, 224, 224)

        print("shape", shape)

        # initialize hooks and single-precision model
        teacher_model = teacher_model.cuda()
        teacher_model = teacher_model.eval()

        # Determine number of classes from model output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, *shape[1:]).cuda()
            dummy_output = teacher_model(dummy_input)
            self.num_classes = dummy_output.shape[1]
            print(f"Model output dimension: {self.num_classes} classes")

        refined_gaussian = []
        labels_list = []

        CE_loss = nn.CrossEntropyLoss(reduction='none').cuda()
        MSE_loss = nn.MSELoss().cuda()

        # hooks, hook_handles = [], []
        for n, m in teacher_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)

        for i in range(num_data//batch_size):
            # initialize the criterion, optimizer, and scheduler

            # 이미지 크기 기반으로 transform 설정
            if hasattr(teacher_model, 'img_size'):
                if teacher_model.img_size == 32:
                    RRC = transforms.RandomResizedCrop(size=32,scale=(augMargin, 1.0))
                elif teacher_model.img_size == 28:
                    RRC = transforms.RandomResizedCrop(size=28,scale=(augMargin, 1.0))
                else:
                    RRC = transforms.RandomResizedCrop(size=224,scale=(augMargin, 1.0))
            else:
                RRC = transforms.RandomResizedCrop(size=224,scale=(augMargin, 1.0))
            RHF = transforms.RandomHorizontalFlip()

            # gaussian_data = torch.randn(shape).cuda()/2.0
            if init_dataset is not None:
                indices = torch.randint(0, init_len, (batch_size,))
                imgs = [init_dataset[idx][0] for idx in indices]
                gaussian_data = torch.stack(imgs).cuda()
            else:
                gaussian_data = torch.randn(shape).cuda()/5.0            
            gaussian_data.requires_grad = True
            optimizer = optim.Adam([gaussian_data], lr=0.5)
            # optimizer = optim.Adam([gaussian_data], lr=0.005)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             min_lr=1e-4,
                                                             verbose=False,
                                                             patience=50)

            # Generate labels based on the actual number of classes
            labels = torch.randint(0, self.num_classes, (len(gaussian_data),)).cuda()
            labels_mask = F.one_hot(labels, num_classes=self.num_classes).float()
            gt = labels.data.cpu().numpy()

            for it in range(500*2):
                # 이미지 크기 기반으로 augmentation 적용
                if hasattr(teacher_model, 'img_size'):
                    if teacher_model.img_size in [28, 32]:
                        # 28x28이나 32x32 모델은 augmentation 없이 처리
                        new_gaussian_data = []
                        for j in range(len(gaussian_data)):
                            new_gaussian_data.append(gaussian_data[j])
                        new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                    else:
                        # 224x224 모델은 augmentation 적용
                        if random.random() < 0.5:
                            new_gaussian_data = []
                            for j in range(len(gaussian_data)):
                                new_gaussian_data.append(RHF(RRC(gaussian_data[j])))
                            new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                        else:
                            new_gaussian_data = []
                            for j in range(len(gaussian_data)):
                                new_gaussian_data.append(gaussian_data[j])
                            new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                else:
                    # 기본적으로 224x224 모델로 처리
                    if random.random() < 0.5:
                        new_gaussian_data = []
                        for j in range(len(gaussian_data)):
                            new_gaussian_data.append(RHF(RRC(gaussian_data[j])))
                        new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                    else:
                        new_gaussian_data = []
                        for j in range(len(gaussian_data)):
                            new_gaussian_data.append(gaussian_data[j])
                        new_gaussian_data = torch.stack(new_gaussian_data).cuda()

                self.mean_list.clear()
                self.var_list.clear()
                self.teacher_running_mean.clear()
                self.teacher_running_var.clear()

                output = teacher_model(new_gaussian_data)
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                a = F.softmax(output, dim=1)
                mask = torch.zeros_like(a)
                b=labels.unsqueeze(1)
                mask=mask.scatter_(1,b,torch.ones_like(b).float())
                p=a[mask.bool()]

                # loss_target = beta * ((1-p).pow(gamma) * CE_loss(output, labels)).mean()
                p_clamped = p.clamp(max=1.0 - 1e-7) # p의 최댓값을 1보다 아주 약간 작게 제한

                if gamma == 0:
                    loss_target = beta * (CE_loss(output, labels)).mean()
                else:
                    # 안정화된 p_clamped 값을 사용하여 loss 계산
                    loss_target = beta * ((1 - p_clamped).pow(gamma) * CE_loss(output, labels)).mean()                
                

                mean_loss = torch.zeros(1).cuda()
                var_loss = torch.zeros(1).cuda()
                for num in range(len(self.mean_list)):
                    mean_loss += MSE_loss(self.mean_list[num], self.teacher_running_mean[num].detach())
                    var_loss += MSE_loss(self.var_list[num], self.teacher_running_var[num].detach())

                num_bn = len(self.mean_list)
                if num_bn == 0:
                    print("Warning: no BatchNorm hooks triggered; skipping BN losses.")
                    total_loss = loss_target
                else:
                    mean_loss = mean_loss / num_bn
                    var_loss = var_loss / num_bn
                    total_loss = mean_loss + var_loss + loss_target
                print(f"Batch: {i}, Iter: {it}, LR: {optimizer.state_dict()['param_groups'][0]['lr']:.4f}, "
                      f"Mean Loss: {mean_loss.item():.4f}, Var Loss: {var_loss.item():.4f}, "
                      f"Target Loss: {loss_target.item():.4f}")

                optimizer.zero_grad()
                # update the distilled data
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(gaussian_data, max_norm=1.0)
                optimizer.step()
                scheduler.step(total_loss.item())

            with torch.no_grad():
                output = teacher_model(gaussian_data.detach())
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                print('d_acc', d_acc)

            refined_gaussian.append(gaussian_data.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

            gaussian_data = gaussian_data.cpu()
            del gaussian_data
            del optimizer
            del scheduler
            del labels
            torch.cuda.empty_cache()

        with open(data_path, "wb") as fp:  # Pickling
            pickle.dump(refined_gaussian, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_path, "wb") as fp:  # Pickling
            pickle.dump(labels_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()
        # return refined_gaussian

    def getDistilData_dsv(self,
                          model_name="resnet18",
                          teacher_model=None,
                          num_data=1280,
                          batch_size=256,
                          num_batch=1,
                          group=1,
                          beta=1.0,
                          gamma=0.0,
                          save_path_head="",
                          init_data_path=None,
                          steps=200):
        """Generate data using Deep Support Vector synthesis."""

        data_path = os.path.join(save_path_head, model_name + "_dsv_beta" + str(beta) + "_group" + str(group) + ".pickle")
        label_path = os.path.join(save_path_head, model_name + "_labels_dsv_beta" + str(beta) + "_group" + str(group) + ".pickle")

        check_path(data_path)
        check_path(label_path)

        # Prepare dataset for initialization if provided
        init_dataset = None
        if init_data_path is not None:
            if hasattr(teacher_model, 'img_size') and teacher_model.img_size == 32:
                init_transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                ])
            else:
                init_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])
            init_dataset = datasets.ImageFolder(init_data_path, transform=init_transform)
            init_len = len(init_dataset)

        # Determine input shape
        if hasattr(teacher_model, 'img_size') and teacher_model.img_size == 32:
            shape = (batch_size, 3, 32, 32)
        else:
            shape = (batch_size, 3, 224, 224)

        teacher_model = teacher_model.cuda().eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, *shape[1:]).cuda()
            dummy_output = teacher_model(dummy_input)
            self.num_classes = dummy_output.shape[1]

        refined_data = []
        labels_list = []

        for i in range(num_data // batch_size):
            if init_dataset is not None:
                indices = torch.randint(0, init_len, (batch_size,))
                imgs = [init_dataset[idx][0] for idx in indices]
                x = torch.stack(imgs).cuda()
            else:
                x = torch.randn(shape).cuda()
            x.requires_grad = True

            lamb = torch.ones(batch_size, device='cuda', requires_grad=True)
            labels = torch.randint(0, self.num_classes, (batch_size,), device='cuda')

            optimizer = optim.Adam([x, lamb], lr=0.01)

            for it in range(steps):
                aug_x = x
                output = teacher_model(aug_x)
                preds = output.argmax(dim=1)

                ce = F.cross_entropy(output, labels, reduction='none')
                primal_loss = (ce * (preds != labels).float()).mean()

                grads_sum = [torch.zeros_like(p) for p in teacher_model.parameters()]
                for b in range(batch_size):
                    loss_b = F.cross_entropy(output[b:b+1], labels[b:b+1])
                    grads_b = torch.autograd.grad(loss_b, list(teacher_model.parameters()), retain_graph=True, create_graph=True)
                    for g_idx, g in enumerate(grads_b):
                        grads_sum[g_idx] += lamb[b] * g

                stat_loss = 0.0
                for p, g in zip(teacher_model.parameters(), grads_sum):
                    stat_loss = stat_loss + (p.detach() + g).abs().mean()

                tv_loss = (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()
                norm_loss = x.pow(2).mean()

                total_loss = stat_loss + beta * primal_loss + 0.001 * tv_loss + 0.001 * norm_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                lamb.data.clamp_(min=0)

            refined_data.append(x.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())

            del x
            del lamb
            del optimizer
            torch.cuda.empty_cache()

        with open(data_path, "wb") as fp:
            pickle.dump(refined_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_path, "wb") as fp:
            pickle.dump(labels_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()


