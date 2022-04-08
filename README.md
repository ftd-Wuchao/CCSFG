# Camera-Conditioned Stable Feature Generation for Isolated Camera Supervised Person Re-IDentification （CVPR'22）

This repository is Pytorch code for our proposed Camera-Conditioned Stable Feature Generation (CCSFG). 

Paper link: https://arxiv.org/abs/2203.15210

## Environment

The code is based on [fastreid](https://github.com/JDAI-CV/fast-reid). See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).

## Dataset Preparation

1. Download Market-1501 and MSMT17.
2. Split Market-1501 and MSMT17 to Market-SCT and MSMT-SCT according to the file names in the market_sct.txt and msmt_sct.txt.
3. ```vim fastreid/data/build.py``` change the ```_root``` to your own data folder.
4. Make new directories in datasets and organize them as follows:
<pre>
+-- datasets
|   +-- market
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
|   +-- msmt
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
</pre>

## Train and test
To train and test the model, you can use following command:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Msmt/AGW_R50.yml
```

If you only want to test the model, you can download our model [Google Drive (Waiting)]() or [Baidu Drive (Code:0000)](https://pan.baidu.com/s/1CJ3aI58R7LZnShkru2Myfg) to ```logs/``` and use following command:
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --config-file configs/Msmt/AGW_R50.yml --eval-only  MODEL.WEIGHTS logs/msmt.pth
```

## Citation
If you find this code useful, please kindly cite the following paper:
<pre>
@article{wu2022camera,
  title={Camera-Conditioned Stable Feature Generation for Isolated Camera Supervised Person Re-IDentification},
  author={Wu, Chao and Ge, Wenhang and Wu, Ancong and Chang, Xiaobin},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
</pre>





