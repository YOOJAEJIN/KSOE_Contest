## 3. 성능 평가 검증을 위한 실행 가이드
### 1. 데이터 전처리
#### 1) 전처리_1.py 실행
전처리는 두가지 단계를 거쳐서 진행합니다. 첫번째 전처리 단계는 전처리_1.py 파일을 실행함으로써 가능하며 각각의 xml 파일을 npy파일로 변경해주는 작업입니다. 이 파일을 실행하기 위해서는 다음과 같은 파이썬 라이브러리가 필요합니다. 
```Python
import os
import base64
import xmltodict
import array
import numpy as np
```
우선 xml 형태의 원본 데이터가 저장된 경로를 지정합니다. train_abnormal_dir, train_normal_dir, val_abnormal_dir, val_normal_dir은 각각 부정맥 학습 데이터, 정상 학습 데이터, 부정맥 검증 데이터, 정상 검증 데이터가 있는 폴더 경로입니다. 그리고 npy 파일로 바뀌어 저장이 가능한 폴더를 생성하여 경로(train_abnormal_dir_save,  train_normal_dir_save, val_abnormal_dir_save, val_normal_dir_save)를 지정합니다. 테스트를 수행할 경우 학습과 검증 데이터 경로 대신, 부정맥 테스트 데이터와 정상 테스트 데이터가 포함된 경로를 각각 설정해주어 전처리를 진행하면 됩니다. 학습시 6으로 시작하는 폴더 내의 데이터는 12-leads ECG 데이터 셋이지만, 5와 8로 시작되는 폴더와 데이터 형태를 맞춰주기 위해 8-leads 데이터 셋만 사용했습니다. 

```Python
#raw data
train_abnormal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/train/arrhythmia/'
train_normal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/train/normal/'
val_abnormal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/validation/arrhythmia/'
val_normal_dir = 'C:/Users/SPS/Desktop/심전도 공모전/electrocardiogram/data/validation/normal/'
raw_data_dir_list = [train_abnormal_dir, train_normal_dir, val_abnormal_dir, val_normal_dir]
#save
train_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/arrhythmia/'
train_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/normal/'
val_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/arrhythmia/'
val_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/normal/'
```
#### 2) 전처리_2.py 실행
두번째 전처리 단계는 전처리_2.py 파일을 실행함으로써 가능하며 부정맥 학습 데이터, 정상 학습 데이터, 부정맥 검증 데이터, 정상 검증 데이터들을 하나의 npy파일로 통합해주는 단계입니다. 이 파일을 실행하기 위해서는 다음과 같은 파이썬 라이브러리가 필요합니다. 테스트를 수행할 경우 학습과 검증 데이터 경로 대신 부정맥 테스트 데이터와 정상 테스트 데이터가 포함된 경로를 각각 설정해주어 전처리를 진행하면 됩니다.

```Python
import os
import numpy as np
```
우선 전처리 1단계에서 만든 폴더 경로를 train_abnormal_dir_save, train_normal_dir_save, val_abnormal_dir_save, val_normal_dir_save 변수에 설정해줍니다. 이후 두번째 전처리 파일이 만들어질 폴더를 지정합니다. 코드 상에서는 부정맥 학습 데이터, 정상 학습 데이터, 부정맥 검증 데이터, 정상 검증 데이터를 위해 총 4번 설정을 해야 합니다.
```Python
train_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/arrhythmia/'
train_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/train/normal/'
val_abnormal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/arrhythmia/'
val_normal_dir_save = 'C:/Users/SPS/Desktop/심전도 공모전/preprocessing/validation/normal/'
...
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/train_abnormal.npy', train_abnormal_data_x)
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/train_normal.npy', train_normal_data_x)
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/val_abnormal.npy', val_abnormal_data_x)
np.save('C:/Users/SPS/Desktop/심전도 공모전/preprocessing_2/val_normal.npy', val_normal_data_x)
```

### 2. 모델 테스트
#### 1) HDAI_부정맥_진단_SPSLAB.py 실행
HDAI_부정맥_진단_SPSLAB.py 파일을 실행하여 모델의 테스트를 진행할 수 있습니다. 이 파일을 실행하기 위해서는 다음과 같은 파이썬 라이브러리가 필요합니다.
```Python
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc
```
모델 학습과 검증시에는 전처리 2단계에서 진행했던 파일이 있는 경로 (train_abnormal_dir_save, train_normal_dir_save, val_abnormal_dir_save, val_normal_dir_save)를 지정해주어야 합니다. 테스트를 수행할 경우에는 부정맥 데이터와 정상 데이터의 경로 (test_abnormal_dir_save, test_normal_dir_save)만 지정해주어도 됩니다. 또한 모델 테스트를 진행 할 경우, 꼭 test_mode = True로 변경해주어야 합니다.

```Python
#train data 디렉토리
train_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/train_abnormal.npy'
train_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/train_normal.npy'
#validation data 디렉토리
val_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/val_abnormal.npy'
val_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/val_normal.npy'
# test data 디렉토리
test_abnormal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/test_abnormal.npy'
test_normal_dir_save = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/데이터/데이터/test_normal.npy'

test_mode = True
```
테스트가 끝난 이후 AUC ROC curve의 area를 계산한 plot을 생성하기 위해서 HDAI_부정맥_진단_SPSLAB.py 파일이 있는 경로에 plots 폴더를 생성해줍니다.

![image](https://user-images.githubusercontent.com/30248006/145714495-a8e68d42-1d5b-4949-9c10-9aec5c395d93.png)

모델 학습 시 모델의 가중치를 저장하기 위해 path2weights의 경로를 설정해주어야 하며, 모델 테스트의 경우에는 모델의 가중치를 불러오기 위해 path3weights의 경로만 설정해주어도 됩니다. 이때 .pt의 이름은 바뀌지 않아도 되지만 'model' 이름의 폴더가 있는 경로를 바꾸어주어야 합니다.
```Python
path2weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #모델 가중치 저장 위치
path3weights = 'C:/Users/Woo Young Hwang/Desktop/SPS/외부 활동/대회/경진대회/Heart Disease AI Datathon 2021/코드/model/ECG_model.pt' #모델 가중치 불러오는 위치
```
