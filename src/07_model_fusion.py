# train_models.py - 训练歌词和音频模型
# ================================================================

import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 配置 ========================

MODEL_DIR = 'models'
#LYRICS_DATA = 'processed_chinese_lyrics_full.csv'
LYRICS_DATA ='balanced_multimodal_dataset.csv'
AUDIO_DATA = 'music_emotion_dataset.csv'

def ensure_model_dir():
    """确保模型目录存在"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"创建模型目录: {MODEL_DIR}")

# ======================== 歌词模型训练 ========================

class LyricsModelTrainer:
    """中文歌词情感分析模型训练器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.performance = {}
        
    def load_data(self):
        """加载歌词数据"""
        print("\n" + "="*60)
        print("1. 加载中文歌词数据")
        print("="*60)
        
        try:
            self.data = pd.read_csv(LYRICS_DATA, encoding='utf-8')
            print(f"成功加载 {len(self.data)} 条歌词数据")
            
            # 数据清洗 - 只保留有情感标签的数据
            self.data = self.data.dropna(subset=['emotion_label'])
            print(f"清洗后: {len(self.data)} 条有效数据")
            
            # 分析类别分布
            print("\n类别分布:")
            distribution = self.data['emotion_label'].value_counts()
            for emotion, count in distribution.items():
                print(f"  {emotion}: {count} ({count/len(self.data)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def extract_features(self, text):
        """提取歌词特征"""
        if pd.isna(text) or text == '':
            return np.zeros(10)
        
        # 分词
        words = list(jieba.cut(str(text)))
        
        # 情感词典
        positive_words = ['快乐', '开心', '幸福', '美好', '爱', '喜欢', '温暖', '阳光', 
                         '甜蜜', '浪漫', '欢乐', '愉快', '美丽', '希望', '梦想']
        negative_words = ['悲伤', '痛苦', '难过', '孤独', '寂寞', '失落', '眼泪', '分离',
                         '伤心', '离别', '痛', '哭', '失去', '绝望', '黑暗']
        calm_words = ['平静', '安宁', '宁静', '淡然', '从容', '舒适', '轻松', '安详',
                     '静谧', '恬静', '悠闲', '平和', '安静', '清净']
        energetic_words = ['激情', '热血', '奔跑', '飞翔', '疯狂', '燃烧', '冲动', '狂欢',
                          '热烈', '澎湃', '激动', '兴奋', '活力', '强烈']
        
        # 计算特征
        total_words = len(words) if words else 1
        
        features = [
            len(text),  # 文本长度
            len(words),  # 词数
            len(set(words)),  # 独特词数
            len(set(words)) / total_words,  # 词汇多样性
            sum(1 for w in words if w in positive_words) / total_words,  # 正面词比例
            sum(1 for w in words if w in negative_words) / total_words,  # 负面词比例
            sum(1 for w in words if w in calm_words) / total_words,  # 平静词比例
            sum(1 for w in words if w in energetic_words) / total_words,  # 活力词比例
            np.mean([len(w) for w in words]) if words else 0,  # 平均词长
            sum(1 for w in words if len(w) > 2) / total_words  # 长词比例
        ]
        
        return np.array(features)
    
    def prepare_features(self):
        """准备训练特征"""
        print("\n提取歌词特征...")
        
        X = []
        y = []
        
        for idx, row in self.data.iterrows():
            if pd.notna(row.get('emotion_label')):
                # 尝试多个可能的歌词列名
                lyrics_text = row.get('lyrics', row.get('processed_clean_text', row.get('text', '')))
                features = self.extract_features(lyrics_text)
                
                X.append(features)
                y.append(row['emotion_label'])
        
        X = np.array(X)
        y = self.label_encoder.fit_transform(y)
        
        self.feature_names = [
            'text_length', 'word_count', 'unique_words', 'vocabulary_diversity',
            'positive_ratio', 'negative_ratio', 'calm_ratio', 'energetic_ratio',
            'avg_word_length', 'long_word_ratio'
        ]
        
        print(f"特征矩阵: {X.shape}")
        print(f"标签数量: {len(y)}")
        print(f"情感类别: {list(self.label_encoder.classes_)}")
        
        return X, y
    
    def handle_imbalance(self, X_train, y_train):
        """处理数据不平衡"""
        print("\n处理数据不平衡...")
        
        unique, counts = np.unique(y_train, return_counts=True)
        min_samples = min(counts)
        
        print("训练集类别分布:")
        for label, count in zip(unique, counts):
            emotion = self.label_encoder.inverse_transform([label])[0]
            print(f"  {emotion}: {count} 样本")
        
        if min_samples < 20:
            print("使用随机过采样...")
            ros = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = ros.fit_resample(X_train, y_train)
        elif min_samples >= 6:
            print("使用SMOTE过采样...")
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        else:
            print("样本太少，使用原始数据...")
            X_balanced, y_balanced = X_train, y_train
        
        print(f"平衡后训练集: {len(X_balanced)} 样本")
        return X_balanced, y_balanced
    
    def train(self):
        """训练模型"""
        print("\n" + "="*60)
        print("2. 训练歌词情感分析模型")
        print("="*60)
        
        # 准备数据
        X, y = self.prepare_features()
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 处理不平衡
        X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        
        # 训练多个模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000, random_state=42
            )
        }
        
        best_score = 0
        best_model_name = ''
        
        print("\n训练和评估模型:")
        for name, model in models.items():
            model.fit(X_train_balanced, y_train_balanced)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  {name}: {accuracy:.3f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.model = model
        
        # 详细评估
        y_pred = self.model.predict(X_test)
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"准确率: {best_score:.3f}")
        
        print("\n分类报告:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # 保存性能指标
        self.performance = {
            'accuracy': best_score,
            'test_size': len(X_test),
            'train_size': len(X_train_balanced),
            'model_type': best_model_name,
            'n_features': len(self.feature_names),
            'n_classes': len(self.label_encoder.classes_)
        }
        
        return best_score
    
    def save_model(self):
        """保存模型"""
        ensure_model_dir()
        
        print("\n保存歌词模型...")
        
        # 保存模型组件
        joblib.dump(self.model, os.path.join(MODEL_DIR, 'lyrics_model.pkl'))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'lyrics_scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(MODEL_DIR, 'lyrics_encoder.pkl'))
        
        # 保存特征信息和性能
        with open(os.path.join(MODEL_DIR, 'lyrics_info.pkl'), 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'performance': self.performance,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        print(f"歌词模型已保存到 {MODEL_DIR} 目录")

# ======================== 音频模型训练 ========================

class AudioModelTrainer:
    """音频特征情感分析模型训练器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.performance = {}
    
    def load_data(self):
        """加载音频数据"""
        print("\n" + "="*60)
        print("3. 加载音频特征数据")
        print("="*60)
        
        try:
            self.data = pd.read_csv(AUDIO_DATA)
            print(f"成功加载 {len(self.data)} 条音频数据")
            
            # 分析类别分布
            print("\n类别分布:")
            distribution = self.data['emotion_label'].value_counts()
            for emotion, count in distribution.items():
                print(f"  {emotion}: {count} ({count/len(self.data)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return False
    
    def prepare_features(self):
        """准备音频特征"""
        print("\n准备音频特征...")
        
        # 音频特征列
        self.feature_names = [
            'valence', 'energy', 'danceability', 'tempo',
            'loudness', 'acousticness', 'instrumentalness',
            'speechiness', 'liveness'
        ]
        
        # 提取特征和标签
        X = self.data[self.feature_names].fillna(0).values
        y = self.label_encoder.fit_transform(self.data['emotion_label'])
        
        print(f"特征矩阵: {X.shape}")
        print(f"标签数量: {len(y)}")
        print(f"情感类别: {list(self.label_encoder.classes_)}")
        
        return X, y
    
    def train(self):
        """训练音频模型"""
        print("\n" + "="*60)
        print("4. 训练音频情感分析模型")
        print("="*60)
        
        # 准备数据
        X, y = self.prepare_features()
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 训练多个模型
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=150, max_depth=12, random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=120, learning_rate=0.1, max_depth=6, random_state=42
            )
        }
        
        best_score = 0
        best_model_name = ''
        
        print("\n训练和评估模型:")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  {name}: {accuracy:.3f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.model = model
        
        # 详细评估
        y_pred = self.model.predict(X_test)
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"准确率: {best_score:.3f}")
        
        print("\n分类报告:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n特征重要性 Top 5:")
            for _, row in importance.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # 保存性能指标
        self.performance = {
            'accuracy': best_score,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'model_type': best_model_name,
            'n_features': len(self.feature_names),
            'n_classes': len(self.label_encoder.classes_)
        }
        
        return best_score
    
    def save_model(self):
        """保存模型"""
        ensure_model_dir()
        
        print("\n保存音频模型...")
        
        # 保存模型组件
        joblib.dump(self.model, os.path.join(MODEL_DIR, 'audio_model.pkl'))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, 'audio_scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(MODEL_DIR, 'audio_encoder.pkl'))
        
        # 保存特征信息和性能
        with open(os.path.join(MODEL_DIR, 'audio_info.pkl'), 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'performance': self.performance,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        print(f"音频模型已保存到 {MODEL_DIR} 目录")

# ======================== 可视化 ========================

def plot_results(lyrics_acc, audio_acc):
    """绘制训练结果"""
    plt.figure(figsize=(12, 5))
    
    # 准确率对比
    plt.subplot(1, 2, 1)
    models = ['歌词模型', '音频模型']
    accuracies = [lyrics_acc, audio_acc]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = plt.bar(models, accuracies, color=colors)
    plt.ylabel('准确率')
    plt.title('模型准确率对比')
    plt.ylim([0, 1])
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # 融合权重
    plt.subplot(1, 2, 2)
    total = lyrics_acc + audio_acc
    weights = [lyrics_acc/total, audio_acc/total]
    
    plt.pie(weights, labels=models, colors=colors, autopct='%1.1f%%')
    plt.title('融合权重分配')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_results.png'), dpi=100)
    plt.show()
    
    print(f"\n结果图表已保存到 {MODEL_DIR}/training_results.png")

# ======================== 主函数 ========================

def main():
    """主训练流程"""
    print("="*60)
    print("音乐情感分析系统 - 模型训练")
    print("="*60)
    
    # 确保模型目录存在
    ensure_model_dir()
    
    # 训练歌词模型
    lyrics_trainer = LyricsModelTrainer()
    if lyrics_trainer.load_data():
        lyrics_acc = lyrics_trainer.train()
        lyrics_trainer.save_model()
    else:
        print("歌词数据加载失败，跳过歌词模型训练")
        lyrics_acc = 0
    
    # 训练音频模型
    audio_trainer = AudioModelTrainer()
    if audio_trainer.load_data():
        audio_acc = audio_trainer.train()
        audio_trainer.save_model()
    else:
        print("音频数据加载失败，跳过音频模型训练")
        audio_acc = 0
    
    # 绘制结果
    if lyrics_acc > 0 and audio_acc > 0:
        plot_results(lyrics_acc, audio_acc)
    
    # 总结
    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    print(f"歌词模型准确率: {lyrics_acc:.3f}")
    print(f"音频模型准确率: {audio_acc:.3f}")
    print(f"\n融合权重:")
    if lyrics_acc > 0 and audio_acc > 0:
        total = lyrics_acc + audio_acc
        print(f"  歌词权重: {lyrics_acc/total:.3f}")
        print(f"  音频权重: {audio_acc/total:.3f}")
    print(f"\n模型保存位置: {MODEL_DIR}/")
    print("="*60)
    
    # 列出保存的文件
    print("\n已保存的模型文件:")
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            print(f"  - {file}")

if __name__ == "__main__":
    main()