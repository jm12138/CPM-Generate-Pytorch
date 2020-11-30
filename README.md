# CPM-Generate-Pytorch
本Repo将模型转换为Pytorch单卡/CPU可运行的版本，原Repo https://github.com/TsinghuaAI/CPM-Generate

原项目首页：https://cpm.baai.ac.cn/

原项目介绍文章：https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw

## 项目说明
参考[CPM-Generate](https://github.com/TsinghuaAI/CPM-Generate)、[CPM-LM-TF2](https://github.com/qhduan/CPM-LM-TF2)、[gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch)等项目开发

感谢上述项目的开源代码和模型

## 使用说明
* 克隆本项目代码
```shell
$ git clone https://github.com/jm12138/CPM-Generate-Pytorch
$ cd CPM-Generate-Pytorch
```

* 准备模型
  * 合并版模型暂未上传完成，所以请参考[转换代码](https://github.com/jm12138/CPM-Generate-Pytorch/blob/main/convert.py)使用原版模型自行进行转换合并

* 安装依赖
```
pytorch
sentencepiece 
jieba 
regex
```

* 运行Demo
```shell
$ python demo.py
```

## 引用
> @article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Cao, Jiannan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
