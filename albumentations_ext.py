import numpy as np
import albumentations as A
import random
from albumentations.core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    to_tuple,
)
def AddSaltPepperNoise(raw_img,density):
    img = raw_img.copy()
    h, w, c = img.shape
    Nd = density
    Sd = 1 - Nd
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])
    # ⽣成⼀个通道的mask
    mask = np.repeat(mask, c, axis=2)
    # 在通道的维度复制，⽣成三通道的mask
    img[mask == 0] = 0# 椒

    img[mask == 1] = 255# 盐
    return img
class SaltPepperNoise(ImageOnlyTransform):
    def __init__(self, density = (0.1,0.3), always_apply=False,p=0.5):
        super(SaltPepperNoise, self).__init__(always_apply, p)
        self.density = density

    def apply(self, img, density=0.3, **params):
        return AddSaltPepperNoise(img, density)

    def get_params(self):
        return {"density": random.uniform(self.density[0], self.density[1])}

    def get_transform_init_args_names(self):
        return ("density")
if __name__=='__main__':
    import cv2
    trans = A.Compose([
            # A.GaussNoise(p=0.5),    # 将高斯噪声应用于输入图像。
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.5),
            # A.Cutout(num_holes=5, max_h_size=3, max_w_size=3, fill_value=255, p=0.5),
            SaltPepperNoise(density=(0.05,0.1),p=1.0),
        ])
    a = cv2.imread("/data/L_E_Data/datasets__MEF/data/MEF/BelgiumHouse.png")
    agu = trans(image = a)
    b = agu["image"]
    cv2.imwrite("b.png",b)