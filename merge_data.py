import numpy as np

data_albu=np.load('./data_fgd.npy')
label_albu=np.load('./label_fgd.npy')
data_adv=np.load('./data_adv_3w.npy')
label_adv=np.load('./label_adv_3w.npy')
#
images=np.concatenate((data_albu[:40000],data_adv[:10000]),axis=0)
labels=np.concatenate((label_albu[:40000],label_adv[:10000]),axis=0)
print(images.shape,labels.shape)
np.save('data.npy', images)
np.save('label.npy', labels)