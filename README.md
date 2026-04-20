茁阳 这是电影推荐系统的代码  首先需要将本地的数据集文件movie.csv和rating.csv传到本地的HDFS里面 还需要补充下载numpy库 然后便可运行 
其中：
用户类型,            学术痛点,                采用的推荐算法,                       核心计算逻辑,                                关键工程处理
新用户 (New Users),冷启动问题 (毫无历史数据),基于内容推荐(Content-Based Filtering),向量空间与余弦相似度(Vector Space & Cosine Similarity),交互问卷获取偏好+ 特征向量化
老用户 (Existing Users),矩阵极度稀疏 (Sparsity),协同过滤(Collaborative Filtering),ALS 交替最小二乘法(Matrix Factorization),隐语义模型预测+ 左反连接 (剔除已看)
