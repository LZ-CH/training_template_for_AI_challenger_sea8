

# 天池比赛:安全AI挑战者计划第八期

## 使用方法

该代码是[AAAI2022 安全AI挑战者计划第八期：Data-Centric Robust Learning on ML Models](https://tianchi.aliyun.com/competition/entrance/531939/introduction)的比赛第一赛季的0.97+方案。

获取数据
```
python get_datatset.py
```
训练模型
```
python train.py
```
## 方法
1 采用PGD对抗算法生成图像

2 采用添加高斯噪声、椒盐噪声、随机尺度变换、随机切除的方式进行数据增广

3 采用筛选的策略来进行迭代筛选数据
