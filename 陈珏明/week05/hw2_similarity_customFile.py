import jieba
import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FastTextVectorizer:
    def __init__(self, stop_words_path):
        # 加载停用词表
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            self.stop_words = set(line.strip() for line in f)
        
    def preprocess_text(self, text):
        """文本预处理：分词 + 清洗"""
        words = jieba.lcut(text)
        return ' '.join([w for w in words 
                       if w not in self.stop_words 
                       and len(w) > 1  # 过滤单字
                       and not w.isdigit()])  # 过滤纯数字

    def train_from_file(self, data_path, model_path, 
                       vector_size=100, window=5, min_count=3):
        """
        从原始文本文件训练词向量模型
        :param data_path: 原始文本文件路径
        :param model_path: 模型保存路径
        :param vector_size: 词向量维度
        :param window: 上下文窗口大小
        :param min_count: 最小词频
        """
        # 生成临时预处理文件
        preprocessed_path = r"reprocessed.txt"
        
        with open(data_path, 'r', encoding='utf-8') as fin, \
             open(preprocessed_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                processed = self.preprocess_text(line.strip())
                if len(processed) > 10:  # 过滤过短文本
                    fout.write(processed + '\n')
        
        # 训练fastText模型
        self.model = fasttext.train_unsupervised(
            input=preprocessed_path,
            model='skipgram',  # 使用skip-gram模型
            dim=vector_size,
            ws=window,
            minCount=min_count,
            epoch=50,
            thread=8
        )
        
        # 保存模型
        self.model.save_model(model_path)
        return self.model

    def get_vector(self, word):
        """获取单个词的向量"""
        return self.model.get_word_vector(word)

    def word_similarity(self, word1, word2):
        """计算两个词的余弦相似度"""
        vec1 = self.get_vector(word1).reshape(1, -1)
        vec2 = self.get_vector(word2).reshape(1, -1)
        return cosine_similarity(vec1, vec2)[0][0]

    def most_similar(self, word, topn=10):
        """查找最相似词语"""
        try:
            return self.model.get_nearest_neighbors(word, k=topn)
        except ValueError:
            print(f"词汇'{word}'不存在于词表中")
            return []

if __name__ == '__main__':
    # 初始化向量化器
    ft = FastTextVectorizer(
        stop_words_path=r"week05\stopwords.txt"
    )
    
    # 训练模型（示例使用豆瓣评论数据）
    model = ft.train_from_file(
        data_path=r"week05\douban_comments_fixed.txt",
        model_path=r"week05\homework\fasttext_model.bin",
        vector_size=150,
        window=8,
        min_count=5
    )
    
    # 测试相似度计算
    test_pairs = [
        ('文学', '小说'),
        ('历史', '战争'),
        ('科学', '技术')
    ]
    for w1, w2 in test_pairs:
        similarity = ft.word_similarity(w1, w2)
        print(f"'{w1}'与'{w2}'的相似度: {similarity:.4f}")
    
    # 查找相似词
    query_word = '哲学'
    print(f"\n与'{query_word}'最相关的词语：")
    for score, word in ft.most_similar(query_word):
        print(f"{word}: {score:.4f}")