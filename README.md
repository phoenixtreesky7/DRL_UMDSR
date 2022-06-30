
# Unsupervised Martian Dust Storm Removal via Disentangled Representation Learning

Dong Zhao __1__; Jia Li __1,2,4__; Hongyu Li __1__; Long Xu __3,4__

__1__ State Key Laboratory of Virtual Reality Technology and Systems, School of Computer Science and Engineering, Beihang University, Beijing 100191, China; 

__2__ Key Laboratory of Solar Activity, National Astronomical Observatories, Chinese Academy of Sciences, Beijing 100101, China;

__3__ National Space Science Center, Chinese Academy of Sciences, Beijing 100190, China;

__4__ Peng Cheng Laboratory, Shenzhen 518000, China.


# Abstract 

>Mars exploration has become a hot spot in recent years and is still advancing rapidly. However, Mars has massive dust storms that may cover many areas of the planet and last for weeks or even months. The local/global dust storms are so influential that they can significantly reduce visibility, and thereby the images captured by the cameras on the Mars rover are degraded severely. This work presents an unsupervised Martian dust storm removal network via __disentangled representation learning (DRL)__. The core idea of the DRL framework is to use the content encoder and dust storm encoder to disentangle the degraded images into content features (on domain-invariant space) and dust storm features (on domain-specific space). The dust storm features carry the full dust storm-relevant prior knowledge from the dust storm images. The `cleaned' content features can be effectively decoded to generate more natural, faithful, clear images. __The primary advantages of this framework are twofold. First, it is among the first to perform unsupervised training in Martian dust storm removal with a single image, avoiding the synthetic data requirements. Second, the model can implicitly learn the dust storm-relevant prior knowledge from the real-world dust storm datasets, avoiding the design of the complicated handcrafted priors.__ Extensive experiments demonstrate the DRL framework's effectiveness and show the promising performance of our network for Martian dust storm removal.


# Training

```python
python train.py --name /your-model-name/ --dataroot /your-data-path/ --which_model_netG dr_ca --dh_real --allmodel --batchSize 8 --ngf 32 --norm sswitch --gpu_ids 0,1
```

# Testing

```python
python test.py --name /your-model-name/ --dataroot /your-data-path/ --which_model_netG dr_ca --dh_real --allmodel --batchSize 1 --ngf 32 --norm sswitch --sb --how_many 1000
```
