# SMP-ETST-2018

## 文本溯源方案
### 模型选择
- [x] simhash
    - 用于大型文本对比，在小文本上较为敏感。在本问题上表现不佳。

- [x] sentence2vec
    - 基线模型，KNN 不设定阈值，F1 在0.5左右。
    - [x] 需要对文本数据的进一步预处理。

- [ ] innoplexus

- [ ] [fasttext](https://github.com/facebookresearch/fastText)
    - 更快的文本处理方法。 
- [ ] char2vec 
    - 看重他的细粒度。
- [ ] zhihu [Top 1、](https://github.com/chenyuntc/PyTorchText)[Top 2](https://github.com/Magic-Bubble/Zhihu) 


### 数据处理
- [x] 文本读取不完整
    - 通过整体读取，按长度划分 ID 和内容，得到解决。感谢学长刘详的指点。 
- [x] 个别 ID 异常的自动替换
    - [X] 目前为手动检查。
- [x] 英文大小写、数字的统一化

### 实现层面
- [ ] 类似谷歌 word2vec 的查询模式
    - 即输入一个句子，返回多条与之相似的句子，并给出相似度。


- [x] 一对多的情况
    - 计算点之前的距离、按阈值选取。
    - 在本例中，一对多是不存在的。由于评价指标单一，未说明是对单行的 F1 累加求得。且由评价指标反馈得到，recall 已经够高，目前是精度不够，引入过多不存在的源文。
        - [ ] 通过模型的差异化，组合多个模型。投票得出确定项。 
            - [ ] xgboost、lightgbm label 编码

- [x] KNN 模型阈值设置
    - [x] 根据反推 TP 和 P 的数量。（会过拟合）
        - P 正例约有 444 个。
    - [x] KNN 预测概览均为一，阈值设置无效
- [x] KNN 阈值替代方案：计算两个句子向量的距离

