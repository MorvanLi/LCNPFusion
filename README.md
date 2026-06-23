# LCNPFusion

Codes for ***LCNPFusion: A Learning Coupled Neural P System for Multi-Focus Image Fusion.***


## 🌐 Usage

### ⚙ Network Architecture

Our LCNPFusion is implemented in ``cnp-model.py``.

### 🏊 Training

**1. Virtual Environment**

```
# create virtual environment
conda create -n LCNPFusion python=3.8.10
conda activate LCNPFusion
# select pytorch version yourself
# install LCNPFusion requirements
pip install -r requirements.txt
```

**2. LCNPFusion Training**

Run 

```
python train.py
```

and the trained model is available in ``'./weights/'``.

### 🏄 Testing

**1. Test datasets**

The test datasets used in the paper have been stored in ``'./test_img/Lytro'``, ``'./test_img/MFFW'``, ``'./test_img/Road-MF'`` and ``'./test_img/Real-MFF'`` .

**2. LCNPFusion Testing**

Run 

```
python Inference_2.py
```