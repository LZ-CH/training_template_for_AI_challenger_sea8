import numpy as np
import albumentations as A
import random
random.seed(11037)





trans = A.Compose([
        A.GaussNoise(p=0.5),    # 将高斯噪声应用于输入图像。
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.5),
        A.Cutout(num_holes=5, max_h_size=3, max_w_size=3, fill_value=255, p=0.5),
    ])
def agu():
    images = np.load('data_9680.npy')
    labels = np.load('label_9680.npy')
    newimages = [x for x in images]
    newlabels = [x for x in labels]
    images = [x for x in images]
    labels = [x for x in labels]
    cnt1 = 0
    cnt2 = 0
    # for i in range(10000):
    #     if random.random()<0.5:
    #         newimages[i] = AddSaltPepperNoise(newimages[i],0.3*random.random())
    #     else:
    #         if random.random()<0.5:
    #             cnt2 += 1
    #             agu = trans(image=newimages[i])
    #             newimages[i] = agu['image']
    # for i in range(10000,len(images)):
    #     if random.random()<0.3:
    #         cnt2 += 1
    #         agu = trans(image=newimages[i])
    #         newimages[i] = agu['image']
    print('Agu num:',cnt1,cnt2)
    np.save('data.npy', np.array(newimages))
    np.save('label.npy', np.array(newlabels))
if __name__=='__main__':
    agu()