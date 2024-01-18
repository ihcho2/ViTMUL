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
from models_ViT.ViT_Unlearning import ViT_unlearning
import utils
from optim import create_optimizer


"""
> [Function] Parse the metadata.
* image_age_list[0] = ["F0001_AGE_D_18_a1.jpg"] = "a"
* image_age_list[1] = ["F0001_AGE_D_18_a2.jpg"] = "a"
* image_age_list[2] = ["F0001_AGE_D_18_a3.jpg"] = "a"
* image_age_list[3] = ["F0001_AGE_D_18_a4.jpg"] = "a"
* image_age_list[4] = ["F0001_AGE_D_18_b1.jpg"] = "b"
...
"""
def parsing(meta_data):
    image_age_list = []
    # iterate all rows in the metadata file
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

class Dataset(Dataset):
    def __init__(self, meta_data, image_directory, transform=None, forget=False, retain=False):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # Process the metadata.
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }

        # After training the original model, we will do "machine unlearning".
        # The machine unlearning requires two datasets, ① forget dataset and ② retain dataset.
        # In this experiment, we set the first 1,500 images to be forgotten and the rest images to be retained.
        if forget:
            self.image_age_list = self.image_age_list[:1500]
        if retain:
            self.image_age_list = self.image_age_list[1500:]

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label


def show_images(images, labels, nrow=6, save_path=None):
    n_images = len(images)
    nrows = n_images // nrow + (n_images % nrow > 0)

    fig, axs = plt.subplots(nrows, nrow, figsize=(14.5, 2.3 * nrows), frameon=False)
    axs = axs.flatten() if n_images > 1 else [axs]

    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = axs[idx]
        img_np = img.numpy().transpose((1, 2, 0))
        ax.imshow(img_np)
        ax.axis('off')

        ax.text(5, 5, label, color='white', fontsize=13,  ha='left', va='top',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.1'))

    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def train():
    start_time = time.time()
    print(f'[Epoch: {epoch + 1} - Training]')
    model.train()
    total = 0
    running_loss = 0.0
    running_corrects = 0

    for i, batch in enumerate(train_dataloader):
        imgs, labels = batch
        imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        optimizer.zero_grad()
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total += labels.shape[0]
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        if i % log_step == log_step - 1:
            print(f'[Batch: {i + 1}] running train loss: {running_loss / total}, running train accuracy: {running_corrects / total}')

    print(f'train loss: {running_loss / total}, accuracy: {running_corrects / total}')
    print("elapsed time:", time.time() - start_time)
    return running_loss / total, (running_corrects / total).item()


def adjust_learning_rate(optimizer, learning_rate, epoch):
    lr = learning_rate
    if epoch >= 10:
        lr /= 10
    if epoch >= 20:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def evaluation(model, data_loader, criterion):
    start_time = time.time()
    print(f'[Test]')
    model.eval()
    total = 0
    running_loss = 0.0
    running_corrects = 0
    running_top2_corrects = 0

    for i, batch in enumerate(data_loader):
        imgs, labels = batch
        imgs, labels = imgs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Top-2 accuracy.
            _, top2_preds = outputs.topk(2, dim=1)  # Get the top 2 class indices.
            top2_correct = top2_preds.eq(labels.view(-1, 1).expand_as(top2_preds))
            running_top2_corrects += top2_correct.any(dim=1).sum().item()

        total += labels.shape[0]
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

#         if (i == 0) or (i % log_step == log_step - 1):
#             print(f'[Batch: {i + 1}] running test loss: {running_loss / total}, running test accuracy: {running_corrects / total}, running top-2 accuracy: {running_top2_corrects / total}')

    print(f'test loss: {running_loss / total}, accuracy: {running_corrects / total}, top-2 accuracy: {running_top2_corrects / total}')
    print("elapsed time:", time.time() - start_time)
    return {'Loss': running_loss / total, 'Acc': running_corrects / total, 'Top-2 Acc': running_top2_corrects / total}

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def compute_losses(model, loader):
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for inputs, y in loader:
        targets = y
        inputs, targets = inputs.cuda(), targets.cuda()

        logits = model(inputs)

        losses = criterion(logits, targets).cpu().detach().numpy()
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


def main(args):
    
    # 1. Create dataloaders
    label_to_age = {
    0: "0-6 years old",
    1: "7-12 years old",
    2: "13-19 years old",
    3: "20-30 years old",
    4: "31-45 years old",
    5: "46-55 years old",
    6: "56-66 years old",
    7: "67-80 years old"
    }

    train_meta_data_path = "./custom_korean_family_dataset_resolution_128/custom_train_dataset.csv"
    train_meta_data = pd.read_csv(train_meta_data_path)
    train_image_directory = "./custom_korean_family_dataset_resolution_128/train_images"

    test_meta_data_path = "./custom_korean_family_dataset_resolution_128/custom_val_dataset.csv"
    test_meta_data = pd.read_csv(test_meta_data_path)
    test_image_directory = "./custom_korean_family_dataset_resolution_128/val_images"

    unseen_meta_data_path = "./custom_korean_family_dataset_resolution_128/custom_test_dataset.csv"
    unseen_meta_data = pd.read_csv(unseen_meta_data_path)
    unseen_image_directory = "./custom_korean_family_dataset_resolution_128/test_images"

#     train_transform = transforms.Compose([
#         transforms.Resize(128),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor()
#     ])

#     test_transform = transforms.Compose([
#         transforms.Resize(128),
#         transforms.ToTensor()
#     ])

#     unseen_transform = transforms.Compose([
#         transforms.Resize(128),
#         transforms.ToTensor()
#     ])
    
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
    
    
    train_dataset = Dataset(train_meta_data, train_image_directory, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = Dataset(test_meta_data, test_image_directory, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    unseen_dataset = Dataset(unseen_meta_data, unseen_image_directory, unseen_transform)
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=64, shuffle=False)
    
    iterator = iter(test_dataloader)
    imgs, labels = next(iterator)

#     label_strs = [label_to_age[label.item()] for label in labels[7:19]]

#     show_images(imgs[7:19], label_strs, nrow=6)

    forget_dataset_train = Dataset(train_meta_data, train_image_directory, train_transform, forget=True)
    forget_dataloader_train = DataLoader(forget_dataset_train, batch_size=64, shuffle=True)

    retain_dataset_train = Dataset(train_meta_data, train_image_directory, train_transform, retain=True)
    retain_dataloader_train = DataLoader(retain_dataset_train, batch_size=64, shuffle=True)

    forget_dataset_test = Dataset(train_meta_data, train_image_directory, test_transform, forget=True)
    forget_dataloader_test = DataLoader(forget_dataset_test, batch_size=64, shuffle=False)

    retain_dataset_test = Dataset(train_meta_data, train_image_directory, test_transform, retain=True)
    retain_dataloader_test = DataLoader(retain_dataset_test, batch_size=64, shuffle=False)
    
    print('='*77)
    print('Train dataset size:', len(train_dataset))
    print('Test dataset size:', len(test_dataset))
    print('Forget dataset size:', len(forget_dataset_train))
    print('Retain dataset size:', len(retain_dataset_train))
    print('Unseen dataset size:', len(unseen_dataset))
    print('='*77)
    criterion = nn.CrossEntropyLoss() # Used for the evaluation function.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    final_results = []
    for seed in args.seeds:
        seed_results = {'utility': [],
                        'forgetting': [],
                        'nomus': []}
        print('='*77)
        print(f'seed {seed}')
        
        set_seed(seed)
        
        model = ViT_unlearning(model_type=args.model_type)
        model = model.cuda()
        
        criterion = nn.CrossEntropyLoss()
        
        new_file_num = 0 
        os.makedirs(f'./results', exist_ok = True)
        
        while os.path.exists(f'./results/MUFAC_{args.model_type}_{args.training_type}_seed_{seed}_output_{new_file_num}.tsv'):
            new_file_num += 1
        
        result_file = f'./results/MUFAC_{args.model_type}_{args.training_type}_seed_{seed}_output_{new_file_num}.tsv'
        print('='*77)
        print('results saved at: ', result_file)
        print('='*77)
        
        with open(result_file, 'w') as file:
            save_results = csv.writer(file, delimiter='\n')
            save_results.writerow('')
            
            
        # args.training_type
        print('-'*77)
        if args.training_type.lower() in ['teacher']:
            used_data_loader = train_dataloader
            print(f'Using model type: {args.training_type}')
        elif args.training_type.lower() in ['retrain']:
            used_data_loader = retain_dataloader_train
            print(f'Using model type: {args.training_type}')
        elif args.training_type.lower() in ['finetune', 'cf3']:
            used_data_loader = retain_dataloader_train
            assert args.checkpoint != None
            print(f'Using model type: {args.training_type}')
            
            print(f'Loading checkpoint from {args.checkpoint}')
            model.load_state_dict(torch.load(args.checkpoint))
            test_acc_0 = evaluation(model, test_dataloader, criterion)
            print(f'Checking the test accuracy of the loaded pre-trained model: {100*test_acc_0["Acc"]: .2f}.')
        
            
            
        # args.optimizer
        print('-'*77)
        if args.optimizer.lower() == 'sgd':
            if args.momentum != None:
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
                print(f'Using optimizer: {args.optimizer.lower()} / momentum: {args.momentum}')
            else:
                optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
                print(f'Using optimizer: {args.optimizer.lower()} / momentum: {args.momentum}')
        elif args.optimizer.lower() == 'adam':
            config_optimizer = {'opt': 'adam', 'lr': args.learning_rate, 'weight_decay': 0.02} 
            arg_opt = utils.AttrDict(config_optimizer)
            optimizer = create_optimizer(arg_opt, model)
            print(f'Using optimizer: {args.optimizer.lower()}')
        elif args.optimizer.lower() == 'adamw':
            config_optimizer = {'opt': 'adamW', 'lr': args.learning_rate, 'weight_decay': 0.02} 
            arg_opt = utils.AttrDict(config_optimizer)
            optimizer = create_optimizer(arg_opt, model)
            print(f'Using optimizer: {args.optimizer.lower()}')
            
            
        if args.training_type.lower() == 'cf3':
            for name, param in model.named_parameters():
                if "visual_encoder.visual.transformer.resblocks.11." in name or "visual_encoder.visual.transformer.resblocks.10." in name or "visual_encoder.visual.transformer.resblocks.9." in name or name == 'fc.weight' or name == 'fc.bias' or 'visual_encoder.visual.ln_post' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
        
        for name, param in model.named_parameters():
            print(f'name: {name}, requires_grad: {param.requires_grad}')
        
        
        dataloader_iterator = iter(forget_dataloader_train)

        num_epochs = args.num_epochs
        
        best_NoMUS = 0
        best_NoMUS_Util = 0
        best_NoMUS_forget = 0
        best_NoMUS_forget_acc = 0
        best_NoMUS_epoch = 0
        tie = False
        
        best_test_acc = 0
        
        for epoch in range(num_epochs):
            model.train()
            
            current_time = time.time()
            running_loss = 0
            global_step = 0
            
            if args.manual_decay:
                print('-'*77)
                print('Using manual decay')
                print('-'*77)
                if epoch >= 10 and epoch < 20:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.learning_rate / 10

                elif epoch >= 20:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.learning_rate / 100
                    
                    
            for batch_idx, (x_, y_) in enumerate(tqdm(used_data_loader, 
                                                      desc=f"{args.model_type}-{args.training_type} (Seed: {seed}, Epoch: {epoch})")):
                y_ = y_.cuda()
                
                outputs_ = model(x_.cuda())
                loss_ = criterion(outputs_, y_.cuda())

                # Overall loss
                joint_loss = loss_
                
                optimizer.zero_grad()
                joint_loss.backward()
                optimizer.step()

                running_loss += joint_loss.item() * x_.size(0)
                    
            average_epoch_loss = running_loss / (len(train_dataloader) * x_.size(0))
            print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")

            # Performance
            test_acc = evaluation(model, test_dataloader, criterion)
            unseen_acc = evaluation(model, unseen_dataloader, criterion)
            forget_acc = evaluation(model, forget_dataloader_test, criterion)
            mia = cal_mia(model.cuda(), forget_dataloader_test, unseen_dataloader)
            model.train()
            
            if 100*(test_acc['Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2 > best_NoMUS:
                best_NoMUS = 100*(test_acc['Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2
                best_NoMUS_Util = 100*test_acc['Acc']
                best_NoMUS_forget = 100*mia['Forgeting Score']
                best_NoMUS_forget_acc = 100*forget_acc['Acc']
                tie = False
                best_NoMUS_epoch = epoch
            elif 100*(test_acc['Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2 == best_NoMUS:
                tie = True
                
            print('-'*77)
            print(f'Seed {seed}, Epoch: {epoch}')
            print(f"Utility score: {100*test_acc['Acc']: .2f}")
            print(f"Forget score: {100*mia['Forgeting Score']: .2f}")
            print(f"Forget-set accuracy: {100*forget_acc['Acc']: .2f}")
            print(f"NoMUS score: {100*(test_acc['Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2 : .2f}")
            print('-'*77)
            print(f"Best NoMUS score so far: {best_NoMUS: .2f} / Util: {best_NoMUS_Util: .2f} / Forget: {best_NoMUS_forget : .2f} / Forget Acc: {best_NoMUS_forget_acc : .2f}")
            print(f"Best NoMUS epoch: {best_NoMUS_epoch}")
            print(f"Exists tie: {tie}")
            print('-'*77)
            if test_acc['Acc'] > best_test_acc:
                best_test_acc = test_acc['Acc'] 
                if args.training_type.lower() == 'teacher' and args.save_teacher == True:
                    torch.save(model.state_dict(), f'MUFAC_{args.model_type}_teacher_seed_{seed}.pth')
            print(f'Best test acc so far: {100*best_test_acc: .2f}')
            print('-'*77)
            
            seed_results['utility'].append(100*test_acc['Acc'])
            seed_results['forgetting'].append(100*mia['Forgeting Score'])
            seed_results['nomus'].append(100*(test_acc['Acc'] + (1 - abs(mia['MIA'] - 0.5) * 2)) / 2)
            
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
        print('-'*77)
        print('-'*77)
        print(f'Best test acc so far: {100*best_test_acc: .2f}')
        print('-'*77)
        print('-'*77)
    epoch_results = [{'utility': [0.0]*len(args.seeds), 
                     'forgetting': [0.0]*len(args.seeds),
                     'nomus': [0.0]*len(args.seeds)} for _ in range(args.num_epochs)]
    
    for j, item in enumerate(final_results):
        for ep in range(args.num_epochs):
            epoch_results[ep]['utility'][j] = round(item['utility'][ep], 2)
            epoch_results[ep]['forgetting'][j] = round(item['forgetting'][ep], 2)
            epoch_results[ep]['nomus'][j] = round(item['nomus'][ep], 2)
            
            
    print('='*77)
    print('Results per epoch: ')
    for j in range(args.num_epochs):
        print('-'*77)
        print(f'epoch {j} results')
        print(f'utility: {epoch_results[j]["utility"]}, avg: {round(sum(epoch_results[j]["utility"])/len(epoch_results[j]["utility"]), 2)}, std: {np.std(epoch_results[j]["utility"], ddof =1)}')
        print(f'forgetting: {epoch_results[j]["forgetting"]}, avg: {round(sum(epoch_results[j]["forgetting"])/len(epoch_results[j]["forgetting"]), 2)}, std: {np.std(epoch_results[j]["forgetting"], ddof =1)}')
        print(f'nomus: {epoch_results[j]["nomus"]}, avg: {round(sum(epoch_results[j]["nomus"])/len(epoch_results[j]["nomus"]), 2)}, std: {np.std(epoch_results[j]["nomus"], ddof =1)}')
        
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
    parser.add_argument('--save_teacher', default = False, type = boolean_string)
    parser.add_argument('--cf3_top_n', default = 3, type = int)
    
    
    parser.add_argument('--seeds',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default = None,
                        help='random seeds')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--learning_rate', default = 0.001, type = float)
    parser.add_argument('--momentum', default=None, type = float)
    
    args = parser.parse_args()

    main(args)
