# 补全RAG model测试代码

将句子转化为向量使用的是multilingual BERT model，如下
model = SentenceTransformer('distiluse-base-multilingual-cased')

`custom_test.py`中包含补全的代码

通过句子相似度计算进行排序完成

## 针对custom_top10
直接运行'score.py'也可以获得结果，使用distiluse-base-multilingual-cased的召回结果为
top 1 recall: 0.37623762376237624
top 3 recall: 0.5346534653465347
top 5 recall: 0.6336633663366337
top 10 recall: 0.7475247524752475

## 针对check_information

这里设置当COS相似度大于0.5即视为包含了相同的信息，可以根据实际情况修改该超参数

## 针对get_answer

将相似度最大的结果输出

## 这里没有实际使用LLM进行测试
