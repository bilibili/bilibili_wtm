# bilibili_wtm

## This is an early version of watermarking method from Bilibili, resisting image and video compression attack well.

### Train
1. Prepare data: download ImageNet2012 validation dataset or use your own dataset, and split it according to image id, i.e., train (1-10000) / validation (10001-15000 / test (15001-25000).
2. ```
   cd bilibili_wtm
   pip install -r requirements.txt
   mkdir results
   ```
3. Pretrain:
   ```
   python train.py --cfg_file option/psnr_tar40_jpgx2.json
   ```
   Note 1: replace "val_path" in cfg_file with your own path. You can extract 500 frames from 1080p video and save them as frames-xxxx.png (xxxx is 0000,0001,0002,...,0498,0499, and resize them to 1920x1072) in order.

   Note 2: choose the model according to the validation recall on the dataset mentioned in Note 1.
4. Finetune with LPIPS loss to reduce visual artifacts:
   ```
   python train.py --cfg_file option/psnr_tar40_jpgx2_lpips.json
   ```

### Test
1. Test with your trained model:
   ```
   python test.py --cfg_file test_setting.json
   ``` 

----
The code is modified from [MBRS](https://github.com/jzyustc/MBRS).
