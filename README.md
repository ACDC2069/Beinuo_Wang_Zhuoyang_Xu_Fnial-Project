茁阳 这是电影推荐系统的代码  首先需要将本地的数据集文件movie.csv和rating.csv传到本地的HDFS里面 还需要补充下载numpy库 然后便可运行 
其中：
1. 新用户：基于内容的推荐 (Content-Based)
底层逻辑：既然没有用户的历史打分记录，就直接让用户显式表达他们的偏好。系统将用户喜欢的“锚点电影”和流派转化为数学特征。

计算过程：

特征向量化 (Vectorization)：利用 CountVectorizer 将文字流派标签转化为多维稀疏向量（One-Hot 编码）。

召回粗筛 (Recall)：利用年代（Year）、热度（Num_ratings）等客观硬指标，快速缩小几十万部电影的候选池。

精排打分 (Ranking)：计算候选电影向量与锚点电影向量在空间中的余弦相似度 (Cosine Similarity)，夹角越小越相似。按相似度降序输出最终列表。

2. 老用户：基于 ALS 矩阵分解的协同过滤 (Collaborative Filtering)
底层逻辑：利用用户既往的评分记录（群体智慧），自动挖掘用户口味和电影属性之间的隐藏联系，预测用户对未看电影的评分。

计算过程：

矩阵分解 (Matrix Factorization)：使用 ALS 算法 将庞大且稀疏的“用户-电影”评分矩阵，拆解为低维度的“用户特征矩阵”和“电影特征矩阵”。

隐语义预测 (Latent Prediction)：将两个特征矩阵重新相乘，计算出该用户对全库所有未评分电影的“预测得分（Predicted Rating）”。

历史掩码 (Interaction Masking)：在工程层面执行 Left Anti Join（左反连接），将预测列表与用户的“已看黑名单”进行比对，严格剔除用户已经评价过的电影，最后按预测得分降序输出。
