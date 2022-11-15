# PointPainting

### Installation 

```pip install timm==0.3.2```
```CUDA 10.1``` and  ```pytorch 1.7.1``` 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```

## Run Instructions
```
python point_paint.py $PATH_TO_DATA $PATH_TO_CONFIG $PATH_TO_CHECKPOINT --device cuda:0 --palette cityscapes
```

```
python point_paint.py ./SegFormer local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py 
                ./SegFormer/segformer.b5.1024x1024.city.160k.pth --device cuda:0 --palette cityscapes
```
