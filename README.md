# Vision Transformer and quantum Vision Transformer
这个文档分为两个部分。前半部分是解析讲解vision transformer的架构，后半部分讲如何用量子线路去构造transformer架构中的自注意力机制。

### Vision Transformer
Vision Transformer 架构里只含有encoder模块，主要模型框架封装在 vit.py 程序中，ViT_train.py 中包含一个训练实例：cifar10数据集，ViT_test.py是相对应的模型测试集合数据。此版本的Vision Transformer是一个简易版对初学者友好的模型架构，在此基础上，大家可以根据自己的需求调整其中单元的结构组成。

### qVision Transformer 
量子版本的vision transformer想要用量子线路去替换经典vision transformer中的自注意力机制。这一块涉及到一些量子计算中基础知识。
