import argparse
import os
import csv
import time
import random
import glob
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import linear_model, model_selection

from torchvision import datasets, transforms, models
from copy import deepcopy
import random
from tqdm import tqdm

from models_ViT.visual_transformers import initialize_clip
from models_ViT.ViT_Unlearning_MUCAC import ViT_unlearning_MUCAC
import utils
from optim import create_optimizer


path = "./"

identities = {}

# (Image File Name, Subject Identity Information)
with open(path + 'CelebA-HQ-identity.txt') as f:
    lines = f.readlines()
    for line in lines:
        file_name, identity = line.strip().split()
        identities[file_name] = identity

print(f'There are {len(set(identities.values()))} identities.')
print(f'There are {len(identities.keys())} images.')

attribute_path = path + 'CelebA-HQ-attribute.txt'

# Create a dictionary for mapping attributes.
attributes_map = {
    "gender": 21,
    "smiling": 32,
    "young": 40
}

# Initialize a dictionary to store the results.
label_map = {}

with open(attribute_path) as f:
    lines = f.readlines()
    for line in lines[2:]:
        splited = line.strip().split()
        file_name = splited[0]
        label_map[file_name] = {attr: int(splited[idx]) for attr, idx in attributes_map.items()}

print(f'There are {len(label_map.keys())} images.')

sample_key = list(label_map.keys())[0]
print(f'Sample labels for {sample_key}: {label_map[sample_key]}')

source_root = path + 'CelebAMask-HQ/CelebA-HQ-img/'

train_index = 190
retain_index = 1250
unseen_index = 4855

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity >= train_index and identity < unseen_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []
        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity < train_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling,  young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

class ForgetDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity >= train_index and identity < retain_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

class RetainDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity < unseen_index and identity >= retain_index:
                gender = int(label_map[file_name]["gender"])
                if gender == -1: gender = 0
                smiling = int(label_map[file_name]["smiling"])
                if smiling == -1: smiling = 0
                young = int(label_map[file_name]["young"])
                if young == -1: young = 0
                self.labels.append((gender, identity, smiling, young))
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

class UnseenDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.image_path_list = glob.glob(os.path.join(source_root, "*"))
        self.transform = transform

        self.image_paths = []
        self.labels = []
        for image_path in self.image_path_list:
            file_name = image_path.split('/')[-1]
            identity = int(identities[file_name])
            if identity < unseen_index:
                continue
            gender = int(label_map[file_name]["gender"])
            if gender == -1: gender = 0
            smiling = int(label_map[file_name]["smiling"])
            if smiling == -1: smiling = 0
            young = int(label_map[file_name]["young"])
            if young == -1: young = 0
            self.labels.append((gender, identity, smiling, young))
            self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        gender = torch.tensor(label[0])
        identity = torch.tensor(label[1])
        smiling = torch.tensor(label[2])
        young = torch.tensor(label[3])

        return image, (gender, identity, smiling, young)

def label_to_string(gender, smiling, young):
    gender_str = "male" if gender == 1 else "female"
    smiling_str = "smiling" if smiling == 1 else "unsmiling"
    young_str = "young" if young == 1 else "old"
    return f"{gender_str}, {smiling_str}, {young_str}"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Define a custom head for the multi-label classification.
class MultiLabelHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(MultiLabelHead, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)

@torch.no_grad()
def evaluation(model, data_loader):
    model.eval()

    running_loss_gender = 0.
    running_corrects_gender = 0

    running_loss_smiling = 0.
    running_corrects_smiling = 0

    running_loss_young = 0.
    running_corrects_young = 0

    for inputs, (gender, identity, smiling, young) in data_loader:
        inputs = inputs.cuda()
        gender = gender.cuda()
        smiling = smiling.cuda()
        young = young.cuda()

        outputs = model(inputs)

        probs = torch.sigmoid(outputs)
        outputs_gender = probs[:, 0]
        outputs_smiling = probs[:, 1]
        outputs_young = probs[:, 2]

        preds_gender = (outputs_gender > 0.5).long()
        preds_smiling = (outputs_smiling > 0.5).long()
        preds_young = (outputs_young > 0.5).long()

        running_corrects_gender += torch.sum(preds_gender == gender)
        running_corrects_smiling += torch.sum(preds_smiling == smiling)
        running_corrects_young += torch.sum(preds_young == young)

    epoch_acc_gender = running_corrects_gender.item() / len(data_loader.dataset)
    epoch_acc_smiling = running_corrects_smiling.item() / len(data_loader.dataset)
    epoch_acc_young = running_corrects_young.item() / len(data_loader.dataset)

    avg_accuracy = (epoch_acc_gender + epoch_acc_smiling + epoch_acc_young) / 3

    return {
        'Average Acc': avg_accuracy,
        'Gender Acc': epoch_acc_gender,
        'Smiling Acc': epoch_acc_smiling,
        'Young Acc': epoch_acc_young,
    }

def compute_losses(net, loader):
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    all_losses = []

    for inputs, (gender, identity, smiling, young) in loader:
        labels = torch.stack((gender, smiling,  young), dim=1).type(torch.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda()

        logits = net(inputs)

        losses = criterion(logits, labels).mean(dim=1).cpu().detach().numpy()
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def cal_mia(model, forget_dataloader_test, unseen_dataloader):
    forget_losses = compute_losses(model, forget_dataloader_test)
    unseen_losses = compute_losses(model, unseen_dataloader)

    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[: len(unseen_losses)]

    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())

    return {'MIA': mia_scores.mean(), 'Forgeting Score': forgetting_score}

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class Noise(nn.Module):
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

def float_to_uint8(img_float):
    """Convert a floating point image in the range [0,1] to uint8 image in the range [0,255]."""
    img_uint8 = (img_float * 255).astype(np.uint8)
    return img_uint8


class SCRUBTraining:
    def __init__(self, teacher, student, retain_dataloader, forget_dataloader, optimizer):
        self.teacher = teacher
        self.student = student
        self.retain_dataloader = retain_dataloader
        self.forget_dataloader = forget_dataloader

        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_div = DistillKL(4.0)
        
        # args.optimizer
        print('-'*77)
        if optimizer.lower() == 'sgd':
            if args.momentum != None:
                self.optimizer = optim.SGD(self.student.parameters(), lr=args.learning_rate, momentum=args.momentum)
                print(f'Using optimizer: {optimizer.lower()} / momentum: {args.momentum}')
            else:
                self.optimizer = optim.SGD(self.student.parameters(), lr=args.learning_rate)
                print(f'Using optimizer: {optimizer.lower()} / momentum: {args.momentum}')
        elif optimizer.lower() == 'adam':
            config_optimizer = {'opt': 'adam', 'lr': args.learning_rate, 'weight_decay': 0.02} 
            arg_opt = utils.AttrDict(config_optimizer)
            self.optimizer = create_optimizer(arg_opt, self.student)
            print(f'Using optimizer: {optimizer.lower()}')
        elif optimizer.lower() == 'adamw':
            config_optimizer = {'opt': 'adamW', 'lr': args.learning_rate, 'weight_decay': 0.02} 
            arg_opt = utils.AttrDict(config_optimizer)
            self.optimizer = create_optimizer(arg_opt, self.student)
            print(f'Using optimizer: {optimizer.lower()}')
            
    def train_epoch(self):
        self.student.train()
        self.teacher.eval()

        # Function to compute accuracy.
        def compute_accuracy(outputs, labels):
            _, predicted = outputs.max(1)
            total = labels.size(0)
            correct = predicted.eq(labels).sum().item()
            return 100 * correct / total

        total_loss_retain, total_accuracy_retain = 0, 0
        total_loss_forget, total_accuracy_forget = 0, 0

        # Training with retain data
        for i, (inputs, (gender, identity, smiling, young)) in enumerate(tqdm(self.retain_dataloader, 
                                                                              desc=f'SCRUB-retain')):
            inputs = inputs.cuda()
            labels = torch.stack((gender, smiling, young), dim=1).type(torch.FloatTensor).cuda()

            # Forward pass: Student (remove torch.no_grad() block for student)
            self.optimizer.zero_grad()  # Reset gradients to zero before computation
            outputs_student = self.student(inputs)
            probs_student =  torch.sigmoid(outputs_student)

            # Forward pass: Teacher
            with torch.no_grad():
                outputs_teacher = self.teacher(inputs)

            # Compute classification loss
            loss_cls = self.criterion_cls(outputs_student, labels)

            # Compute divergence loss with teacher's outputs
            loss_div_retain = self.criterion_div(outputs_student, outputs_teacher)
            loss = loss_cls + loss_div_retain

            # Backpropagation
            loss.backward()
            self.optimizer.step()

        # Training with forget data.
        for i, (inputs, (gender, identity, smiling, young)) in enumerate(tqdm(self.forget_dataloader, 
                                                                              desc=f'SCRUB-forget')):
            inputs = inputs.cuda()
            labels = torch.stack((gender, smiling, young), dim=1).type(torch.FloatTensor).cuda()
        
            # Forward pass: Student (remove torch.no_grad() block for student)
            self.optimizer.zero_grad()  # Reset gradients to zero before computation
            outputs_student = self.student(inputs)
            probs_student =  torch.sigmoid(outputs_student)
            
            # Forward pass: Teacher
            with torch.no_grad():
                outputs_teacher = self.teacher(inputs)
                
            # We want to maximize the divergence for the forget data.
            loss_div_forget = -args.coefficient*self.criterion_div(outputs_student, outputs_teacher)
            total_loss_forget += loss_div_forget.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_div_forget.backward()
            self.optimizer.step()
        
def main(args):
    if args.transform == 1:
        train_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor()
        ])

        unseen_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor()
        ])
    elif args.transform == 2:
        # From mPLUG
        from torchvision import transforms
        from randaugment import RandomAugment
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        train_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(128,scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])  

        test_transform = transforms.Compose([
            transforms.Resize((128,128),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])  
        unseen_transform = transforms.Compose([
            transforms.Resize((128,128),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])

    train_set = TrainDataset(transform=train_transform)
    test_set = TestDataset(transform=test_transform)
    forget_set_train = ForgetDataset(transform=train_transform)
    forget_set_test = ForgetDataset(transform=test_transform)
    retain_set_train = RetainDataset(transform=train_transform)
    retain_set_test = RetainDataset(transform=test_transform)
    unseen_set = UnseenDataset(transform=unseen_transform)
    
    print('='*77)
    print('Train dataset size:', len(train_set))
    print('Test dataset size:', len(test_set))
    print('Forget dataset size:', len(forget_set_train))
    print('Retain dataset size:', len(retain_set_train))
    print('Unseen dataset size:', len(unseen_set))
    print('='*77)
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    forget_dataloader_train = torch.utils.data.DataLoader(forget_set_train, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    forget_dataloader_test = torch.utils.data.DataLoader(forget_set_test, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    retain_dataloader_train = torch.utils.data.DataLoader(retain_set_train, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    retain_dataloader_test = torch.utils.data.DataLoader(retain_set_test, batch_size=64, shuffle=False, num_workers=2,pin_memory=True)
    unseen_dataloader = torch.utils.data.DataLoader(unseen_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    final_results = []
    for seed in args.seeds:
        seed_results = {'utility': [],
                        'forgetting': [],
                        'nomus': []}
        print('='*77)
        print(f'seed {seed}')
        
        set_seed(seed)
        
        model = ViT_unlearning_MUCAC(model_type=args.model_type)
        model = model.cuda()

        criterion = nn.BCEWithLogitsLoss()
        
        new_file_num = 0 
        os.makedirs(f'./results', exist_ok = True)
        
        while os.path.exists(f'./results/MUCAC_{args.model_type}_{args.training_type}_seed_{seed}_output_{new_file_num}.tsv'):
            new_file_num += 1
        
        result_file = f'./results/MUCAC_{args.model_type}_{args.training_type}_seed_{seed}_output_{new_file_num}.tsv'
        print('='*77)
        print('results saved at: ', result_file)
        print('='*77)
        
        with open(result_file, 'w') as file:
            save_results = csv.writer(file, delimiter='\n')
            save_results.writerow('')
            
        # args.training_type
        print('-'*77)
        assert args.training_type.lower() =='scrub'
        used_data_loader = retain_dataloader_train
        assert args.checkpoint != None
        print(f'Using training type: {args.training_type}')

        print(f'Loading checkpoint from {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint))
        test_acc_0 = evaluation(model, test_dataloader)
        print(f'Checking the test accuracy of the loaded pre-trained model: {100*test_acc_0["Average Acc"]: .2f}.')
        
        # teacher model
        teacher = deepcopy(model)
        teacher = teacher.cuda()
        test_acc_0 = evaluation(teacher, test_dataloader)
        print(f'Checking the test accuracy of the loaded teacher model: {100*test_acc_0["Average Acc"]: .2f}%')
        
        scrub_trainer = SCRUBTraining(teacher, model, retain_dataloader_train, forget_dataloader_train, args.optimizer)
        
        best_NoMUS = 0
        best_NoMUS_Util = 0
        best_NoMUS_forget = 0
        best_NoMUS_forget_acc = 0
        best_NoMUS_epoch=0
        tie = False
        
        best_test_acc = 0
                    
        dataloader_iterator = iter(forget_dataloader_train)
        num_epochs = args.num_epochs
        
        for epoch in range(num_epochs):
            current_time = time.time()
            running_loss = 0
            global_step = 0
            
            if args.manual_decay:
                print('-'*77)
                print('Using manual decay')
                print('-'*77)
                if epoch >= 10 and epoch < 20:
                    for param_group in scrub_trainer.optimizer.param_groups:
                        param_group['lr'] = args.learning_rate / 10

                elif epoch >= 20:
                    for param_group in scrub_trainer.optimizer.param_groups:
                        param_group['lr'] = args.learning_rate / 100
                        
            scrub_trainer.train_epoch()
                        
            # Performance
            test_acc = evaluation(model, test_dataloader)
            unseen_acc = evaluation(model, unseen_dataloader)
            forget_acc = evaluation(model, forget_dataloader_test)
            mia = cal_mia(model.cuda(), forget_dataloader_test, unseen_dataloader)
            model.train()
            
            if 100*(test_acc['Average Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2 > best_NoMUS:
                best_NoMUS = 100*(test_acc['Average Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2
                best_NoMUS_Util = 100*test_acc['Average Acc']
                best_NoMUS_forget = 100*mia['Forgeting Score']
                best_NoMUS_forget_acc = 100*forget_acc['Average Acc']
                tie = False
                best_NoMUS_epoch = epoch
            elif test_acc['Average Acc'] == best_NoMUS:
                tie = True
            
            print('-'*77)
            print(f'Seed {seed}, Epoch: {epoch}')
            print(f"Utility score: {100*test_acc['Average Acc']: .2f}")
            print(f"Forget score: {100*mia['Forgeting Score']: .2f}")
            print(f"Forget-set accuracy: {100*forget_acc['Average Acc']: .2f}")
            print(f"NoMUS score: {100*(test_acc['Average Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2 : .2f}")
            print('-'*77)
            print(f"Best NoMUS score so far: {best_NoMUS: .2f} / Util: {best_NoMUS_Util: .2f} / Forget: {best_NoMUS_forget : .2f} / Forget Acc: {best_NoMUS_forget_acc : .2f}")
            print(f"Best NoMUS epoch: {best_NoMUS_epoch}")
            print(f"Exists tie: {tie}")
            print('-'*77)
            if test_acc['Average Acc'] > best_test_acc:
                best_test_acc = test_acc['Average Acc'] 
                if args.training_type.lower() == 'teacher' and args.save_teacher == True:
                    torch.save(model.state_dict(), f'MUCAC_{args.model_type}_teacher_seed_{seed}.pth')
            print(f'Best test acc so far: {100*best_test_acc: .2f}')
            
            seed_results['utility'].append(100*test_acc['Average Acc'])
            seed_results['forgetting'].append(100*mia['Forgeting Score'])
            seed_results['nomus'].append(100*(test_acc['Average Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2)
            
            # This is just for my personal recording purpose. Not really conducive for reading the results.
            with open(result_file, 'w') as file:
                save_results = csv.writer(file, delimiter='\n')
                for j in range(len(seed_results['utility'])):
                    save_results.writerow([round(seed_results['utility'][j], 2), 
                                           round(seed_results['forgetting'][j], 2), 
                                           round(seed_results['nomus'][j],2)])
                    save_results.writerow(['',''])
            
        final_results.append(seed_results)
        print('='*77)
        print(f'seed {seed} results')
        utility_seed = []
        forgetting_seed = []
        nomus_seed = []
        
        for j in range(len(seed_results["utility"])):
            utility_seed.append(round(seed_results["utility"][j], 2))
            forgetting_seed.append(round(seed_results["forgetting"][j], 2))
            nomus_seed.append(round(seed_results["nomus"][j], 2))
            
        print(f'utility: {utility_seed}')
        print('-'*77)
        print(f'Avg: {round(sum(utility_seed)/len(utility_seed), 3)}, std: {np.std(utility_seed, ddof =1)}')
        print('-'*77)
        print(f'forgetting: {forgetting_seed}')
        print('-'*77)
        print(f'Avg: {round(sum(forgetting_seed)/len(forgetting_seed), 3)}, std: {np.std(forgetting_seed, ddof =1)}')
        print('-'*77)
        print(f'nomus: {nomus_seed}')
        print('-'*77)
        print(f'Avg: {round(sum(nomus_seed)/len(nomus_seed), 3)}, std: {np.std(nomus_seed, ddof =1)}')
        print('='*77)
        print(f"Best NoMUS score so far: {best_NoMUS: .2f} / Util: {best_NoMUS_Util: .2f} / Forget: {best_NoMUS_forget : .2f} / Forget Acc: {best_NoMUS_forget_acc : .2f}")
        print('='*77)

#     epoch_results = [{'utility': [0.0]*len(args.seeds), 
#                      'forgetting': [0.0]*len(args.seeds),
#                      'nomus': [0.0]*len(args.seeds)} for _ in range(args.num_epochs)]

#     for j, item in enumerate(final_results):
#         for ep in range(args.num_epochs):
#             epoch_results[ep]['utility'][j] = round(item['utility'][ep], 2)
#             epoch_results[ep]['forgetting'][j] = round(item['forgetting'][ep], 2)
#             epoch_results[ep]['nomus'][j] = round(item['nomus'][ep], 2)


#     print('Results per epoch: ')
#     for j in range(args.num_epochs):
#         print('-'*77)
#         print(f'epoch {j} results')
#         print(f'utility: {epoch_results[j]["utility"]}, avg: {round(sum(epoch_results[j]["utility"])/len(epoch_results[j]["utility"]), 2)}, std: {np.std(epoch_results[j]["utility"], ddof =1)}')
#         print(f'forgetting: {epoch_results[j]["forgetting"]}, avg: {round(sum(epoch_results[j]["forgetting"])/len(epoch_results[j]["forgetting"]), 2)}, std: {np.std(epoch_results[j]["forgetting"], ddof =1)}')
#         print(f'nomus: {epoch_results[j]["nomus"]}, avg: {round(sum(epoch_results[j]["nomus"])/len(epoch_results[j]["nomus"]), 2)}, std: {np.std(epoch_results[j]["nomus"], ddof =1)}')


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--manual_decay', default = True, type = boolean_string)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--model_type', default='ViT-B-16', type=str)
    parser.add_argument('--training_type', default='teacher', type=str)
    parser.add_argument('--transform', default = 2, type = int)
    parser.add_argument('--save_teacher', default = False, type = boolean_string)
    parser.add_argument('--cf3_top_n', default = 3, type = int)
    parser.add_argument('--coefficient', default=None, type = float)
    
    parser.add_argument('--seeds',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default = None,
                        help='random seeds')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--momentum', default=None, type = float)
    
    args = parser.parse_args()

    main(args)

