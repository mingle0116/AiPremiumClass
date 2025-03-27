import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

class BookRecommender:
    def __init__(self):
        self.books = []
        self.book_names = []
        self.tfidf_matrix = None
        self.vectorizer = TfidfVectorizer()
        self.bm25 = None

    def load_data(self, filename):
        book_comments = {}
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for item in reader:
                book = item['book'].strip()
                comment = item['body'].strip()
                if not book or not comment:
                    continue
                words = jieba.lcut(comment)
                book_comments.setdefault(book, []).extend(words)
        return book_comments

    def prepare_data(self, data_file, stop_words_file):
        # 加载停用词
        stop_words = set(line.strip() for line in open(stop_words_file, 'r', encoding='utf-8'))
        
        # 加载图书数据
        book_data = self.load_data(data_file)
        
        # 准备语料库
        self.book_names = list(book_data.keys())
        corpus = [' '.join(words) for words in book_data.values()]
        
        # 构建TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # 构建BM25
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def recommend(self, query, algorithm='tfidf', top_n=5):
        """双算法推荐核心方法"""
        if algorithm == 'tfidf':
            query_vec = self.vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        elif algorithm == 'bm25':
            tokenized_query = query.split()
            sim_scores = self.bm25.get_scores(tokenized_query)
        else:
            raise ValueError("Unsupported algorithm")

        top_indices = np.argsort(sim_scores)[-top_n:][::-1]
        return [(self.book_names[i], sim_scores[i]) for i in top_indices]

if __name__ == '__main__':
    recommender = BookRecommender()
    recommender.prepare_data(
        data_file=r"c:\Users\Administrator\Desktop\AI_Python\AiPremiumClass\陈珏明\week05\douban_comments_fixed.txt",
        stop_words_file=r"c:\Users\Administrator\Desktop\AI_Python\AiPremiumClass\陈珏明\week05\stopwords.txt"
    )
    
    # 双算法测试
    test_query = "值得反复阅读的经典"
    
    print("\nTF-IDF推荐结果：")
    for book, score in recommender.recommend(test_query):
        print(f"{book} (相似度: {score:.4f})")
        
    print("\nBM25推荐结果：")
    for book, score in recommender.recommend(test_query, algorithm='bm25'):
        print(f"{book} (评分: {score:.1f})")