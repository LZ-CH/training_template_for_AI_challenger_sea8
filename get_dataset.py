import numpy as np
import torchvision
import random
import torch
import os
import random
import shutil

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from config import args_resnet, args_densenet
from utils import load_model, AverageMeter, accuracy
import ssl
from train import *
import albumentations as A
import albumentations_ext as AE
from deeprobust.image.attack.pgd import PGD
from deeprobust.image.attack.onepixel import Onepixel
from deeprobust.image.attack.deepfool import DeepFool
from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.fgsm import FGSM

from deeprobust.image.config import attack_params
import time
ssl._create_default_https_context = ssl._create_unverified_context
# def clip_bound(adv):
#     #裁剪边界
#     mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
#     std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
#     adv = adv * std + mean
#     adv = np.clip(adv, 0., 1.)
#     adv = (adv - mean) / std
#     return adv.float()
def inverse_normal(img_normal):
    #逆归一化
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 3))
    std = np.array([0.2023, 0.1994, 0.2010]).reshape((1, 1, 3))
    img = img_normal * std + mean
    img = np.clip(img, 0., 1.)
    return img



seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trans = A.Compose([
        A.OneOf([
            A.GaussNoise(p=1.0),    # 将高斯噪声应用于输入图像。
            AE.SaltPepperNoise(density=(0.05,0.1),p=1.0),
            # A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.1), always_apply=False, p=1.0),
        ],p=0.75),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.5),
        # A.ElasticTransform(alpha = 1,sigma = 50,alpha_affine = 50,interpolation = 1,border_mode = 4,\
        #     value = None,mask_value = None,always_apply = False,approximate = False,p = 0.5),
        A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
        A.OneOf([
            A.Cutout(num_holes=5, max_h_size=3, max_w_size=3, fill_value=255, p=1.0),
            A.Cutout(num_holes=5, max_h_size=3, max_w_size=3, fill_value=0, p=1.0),
            A.Cutout(num_holes=5, max_h_size=3, max_w_size=3, fill_value=127, p=1.0),
        ],p = 0.5),
    ])

# trans = A.Compose([
#         A.GaussNoise(p=0.3),    # 将高斯噪声应用于输入图像。
#         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.3),
#         A.RandomBrightnessContrast(p=0.3),   # 随机明亮对比度
#         A.RandomGridShuffle(grid=(2,2), p=0.3), #随机网格切换
#         A.OneOf([
#             A.RandomRain (p=0.5),
#             A.RandomShadow(p=0.5),
#             A.RandomSnow(p=0.5),
#             A.RandomSunFlare(p=0.5),
#         ],p = 0.3
#         )
#     ])
   
def get_attack_data(net_list):
    #添加对抗图像
    attack_params['PGD_CIFAR10']['print_process'] =False
    net_list = [net.eval() for net in net_list]

    PGD_adversary_list = [PGD(net) for net in net_list]
    # FGSM_adversary_list = [FGSM(net, device='cuda') for net in net_list]
    # CW_adversary_list = [CarliniWagner(net, device='cuda') for net in net_list]
    # OP_adversary_list = [Onepixel(net) for net in net_list]
    # DeepFool_adadversary_list = [DeepFool(net) for net in net_list]
    n_batchsize = 128
    attack_step = 20
    train_loader  = torch.utils.data.DataLoader(
                    datasets.CIFAR10('./data', train = True, download=False,
                    transform = transform_test),
                    batch_size = n_batchsize, shuffle=True)

    test_loader  = torch.utils.data.DataLoader(
                    datasets.CIFAR10('./data', train = False, download=False,
                    transform = transform_test),
                    batch_size = n_batchsize, shuffle=True)
    images = []
    soft_labels = []
    cnt = 0
    # for loader in [test_loader,train_loader]:
    attack_num = 4
    for an in range(attack_num):
        for loader in [test_loader]:
            for image, label in loader:
                image_tensor = image.cuda()
                label_tensor = label.cuda()
                Adv_img_batch = image_tensor
                t0 = time.time()
                for adversary in PGD_adversary_list:
                    if random.random()<0.8:
                        Adv_img_batch = adversary.generate(Adv_img_batch, label_tensor, num_steps=attack_step,**attack_params['PGD_CIFAR10']).float()
                        Adv_img_batch = Adv_img_batch.detach()
                # for adversary in FGSM_adversary_list:
                #     if random.random()<0.8:
                #         Adv_img_batch = adversary.generate(Adv_img_batch, label_tensor,**attack_params['FGSM_MNIST']).float()
                #         Adv_img_batch = Adv_img_batch.detach()
                # for adversary in OP_adversary_list:
                #     if random.random()<0.8:
                #         Adv_img_batch = adversary.generate(Adv_img_batch, label_tensor).float()
                #         Adv_img_batch = Adv_img_batch.detach()
                for i in range(Adv_img_batch.shape[0]):
                    Adv_img = Adv_img_batch[i,:,:,:]
                    each_label = label[i]
                    Adv_img = Adv_img.squeeze().permute(1,2,0).cpu().numpy()
                    Adv_img = (Adv_img*255).astype('uint8')
                    soft_label = np.zeros(10)
                    soft_label[each_label.item()] = 1

                    # if random.random()<0.5:
                    #     agu = trans(image = Adv_img)
                    #     Adv_img = agu['image']

                    images.append(Adv_img)
                    soft_labels.append(soft_label)
                    cnt += 1
                print(cnt,'/',60000)
                print('time:',(time.time()-t0)/n_batchsize,' s/patch')        
    return images,soft_labels

    
    
def prepare_data_stage1():
    #原始图像
    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    images = []
    soft_labels = []
    other_images = []
    other_labels = []
    for image, label in dataset_test:
        image = np.array(image)
        images.append(image)
        soft_label = np.zeros(10)
        soft_label[label] = 1
        soft_labels.append(soft_label)

    for n, [image, label] in enumerate(dataset_train):
        image = np.array(image)
        other_images.append(image)
        other_label = np.zeros(10)
        other_label[label] = 1 # an unnormalized soft label vector
        other_labels.append(other_label)
    print('Train:',len(images),'Test:',len(other_images))
    np.save('data.npy', np.array(images))
    np.save('label.npy', np.array(soft_labels))
    return images,soft_labels,other_images,other_labels
def prepare_data_stage2():
    #数据增广
    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    dataset_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    images = []
    soft_labels = []
    for dataloader in [dataset_test]:
        for image, label in dataloader:
            image = np.array(image)
            images.append(image)
            soft_label = np.zeros(10)
            soft_label[label] = 1
            soft_labels.append(soft_label)
    newimages = []
    newlabels = []
    num = 1
    for j in range(num):
        for i in images:
            agu = trans(image = i)
            newimages.append(agu['image'])
    newlabels = soft_labels*num
    return newimages,newlabels
def train_epoch(model_list,epochnum):
    args = args_resnet
    # Data
    trainset = MyDataset(transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    # Model
    best_acc = 0  # best test accuracy
    for model in model_list:
        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.cuda()
        # Train and val
        for epoch in tqdm(range(epochnum)):
            train_loss, train_acc = train(trainloader, model, optimizer)
            print(args)
            print('acc: {}'.format(train_acc))
            # save model
            best_acc = max(train_acc, best_acc)
            if args['scheduler_name'] != None:
                scheduler.step()
        print('Best acc:')
        print(best_acc)
def get_data():
    one_stage_epoch = 150
    each_epoch = 20
    # model = load_model('resnet50')
    # model = model.cuda()
    model_list = [load_model('resnet50').cuda(),load_model('densenet121').cuda()]
    # model_list = [torch.nn.DataParallel(net) for net in model_list]
    images,labels,other_images,other_labels = prepare_data_stage1()

    #显存测试
    # train_epoch(model_list,1)
    # newimages,newlabels = get_attack_data(model_list)

    train_epoch(model_list,one_stage_epoch)#预训练

    done = False

    #添加正常的图像
    # flags = np.zeros(len(other_labels))
    # while True:
    #     images,labels,flags,isempty,done = test_add_data(images,labels,other_images,other_labels,model_list,flags,addnum = 2000)
    #     train_epoch(model_list,each_epoch)
    #     if isempty or done:
    #         break
    
    #额外添加图像
    addstep = 0
    while not done:
        #直到添加到5w
        if addstep%2==0:
            newimages,newlabels = get_attack_data(model_list)#生成对抗图像
        else:
            newimages,newlabels = prepare_data_stage2()#生成增广图像
        addstep += 1
        flags = np.zeros(len(newlabels))
        while True:
            images,labels,flags,isempty,done = test_add_data(images,labels,newimages,newlabels,model_list,flags,addnum = 2000,all_error_is_added=True)
            if done:
                break
            train_epoch(model_list,each_epoch)
            if isempty:
                break
    
def test_add_data(images,labels,other_images,other_labels,model_list,flags,addnum = 2000,all_error_is_added = False):
    model_list = [n.eval() for n in model_list]
    count = 0
    isempty = True
    done = False
    for n in range(len(other_images)):
        if len(images) == 50000:
            done = True
            break
        if count == addnum:
            isempty = False
            break
        if flags[n] == 0:#1为无需添加的图像,0为待添加图像 
            flags[n] = 1
            img = Image.fromarray(other_images[n])
            img = transform_test(img)
            img = img.unsqueeze(dim = 0).cuda()
            gt_label = np.argmax(other_labels[n])
            isadd = False
            for model in model_list:
                pred = model(img).detach()
                pred_label = torch.argmax(pred,dim = 1)
                if all_error_is_added:
                    if pred_label != gt_label:
                        isadd = True
                    else:
                        isadd = False
                        break
                else:
                    if pred_label != gt_label:
                        isadd = True 
                        break
            if isadd:
                images.append(other_images[n])
                labels.append(other_labels[n])
                count += 1



    np.save('data.npy', np.array(images))
    np.save('label.npy', np.array(labels))
    print('Train num:',len(images))
    print('Add_num:',np.sum(flags),'Total:',flags.shape[0],'Add rate:',np.sum(flags)/flags.shape[0])
    return images,labels,flags,isempty,done

if __name__ == '__main__':
    get_data()
    main()
