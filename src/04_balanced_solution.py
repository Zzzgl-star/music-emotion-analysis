import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("平衡的多模态音乐情感分析系统")
print("="*70)

class BalancedMultiModalAnalyzer:
    """处理不平衡数据的多模态分析器"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler_text = StandardScaler()
        self.scaler_audio = StandardScaler()
        self.balanced_data = None
        self.models = {}
        
    def load_and_analyze_data(self, lyrics_path, audio_path):
        """加载并分析数据分布"""
        print("\n1. 数据加载与分析")
        print("-"*50)
        
        # 加载数据
        self.lyrics_df = pd.read_csv(lyrics_path)
        self.audio_df = pd.read_csv(audio_path)
        
        print(f"歌词数据集: {len(self.lyrics_df)} 样本")
        print(f"音频数据集: {len(self.audio_df)} 样本")
        
        # 分析分布
        print("\n原始数据分布:")
        print("歌词数据集:")
        lyrics_dist = self.lyrics_df['emotion_label'].value_counts()
        for emotion, count in lyrics_dist.items():
            print(f"  {emotion}: {count} ({count/len(self.lyrics_df)*100:.1f}%)")
        
        print("\n音频数据集:")
        audio_dist = self.audio_df['emotion_label'].value_counts()
        for emotion, count in audio_dist.items():
            print(f"  {emotion}: {count} ({count/len(self.audio_df)*100:.1f}%)")
        
        return self.lyrics_df, self.audio_df
    
    def create_balanced_pairs(self, strategy='smart_pairing'):
        """创建平衡的配对数据集"""
        print("\n2. 创建平衡数据集")
        print("-"*50)
        
        if strategy == 'smart_pairing':
            return self._smart_pairing()
        elif strategy == 'augmentation':
            return self._augmentation_pairing()
        else:
            return self._simple_pairing()
    
    def _smart_pairing(self):
        """智能配对策略：利用音频数据的平衡分布"""
        print("使用智能配对策略...")
        
        paired_data = []
        emotions = ['happy', 'sad', 'calm', 'energetic']
        target_per_class = 400  # 每类400个样本
        
        for emotion in emotions:
            # 音频数据（已平衡）
            audio_emotion = self.audio_df[self.audio_df['emotion_label'] == emotion]
            if len(audio_emotion) > target_per_class:
                audio_emotion = audio_emotion.sample(n=target_per_class, random_state=42)
            
            # 歌词数据（不平衡）
            lyrics_emotion = self.lyrics_df[self.lyrics_df['emotion_label'] == emotion]
            
            # 根据歌词数据量决定策略
            if len(lyrics_emotion) == 0:
                print(f"警告: {emotion} 类别在歌词数据中没有样本")
                continue
            
            if len(lyrics_emotion) < target_per_class:
                # 过采样策略
                if len(lyrics_emotion) < 20:
                    # 极少样本，使用SMOTE前先复制
                    print(f"  {emotion}: {len(lyrics_emotion)} 个样本，使用增强过采样")
                    lyrics_sampled = self._augment_samples(lyrics_emotion, target_per_class)
                else:
                    # 普通过采样
                    print(f"  {emotion}: {len(lyrics_emotion)} 个样本，使用随机过采样")
                    lyrics_sampled = lyrics_emotion.sample(n=target_per_class, replace=True, random_state=42)
            else:
                # 欠采样
                print(f"  {emotion}: {len(lyrics_emotion)} 个样本，使用欠采样")
                lyrics_sampled = lyrics_emotion.sample(n=target_per_class, replace=False, random_state=42)
            
            # 重置索引
            audio_emotion = audio_emotion.reset_index(drop=True)
            lyrics_sampled = lyrics_sampled.reset_index(drop=True)
            
            # 配对
            for i in range(min(len(audio_emotion), len(lyrics_sampled))):
                paired_row = {
                    # 歌词特征
                    'text': lyrics_sampled.iloc[i].get('processed_clean_text', ''),
                    'text_length': lyrics_sampled.iloc[i].get('text_length', 0),
                    'sentiment_polarity': lyrics_sampled.iloc[i].get('sentiment_polarity', 0),
                    'emotion_intensity': lyrics_sampled.iloc[i].get('emotion_intensity', 0),
                    'positive_ratio': lyrics_sampled.iloc[i].get('positive_ratio', 0),
                    'negative_ratio': lyrics_sampled.iloc[i].get('negative_ratio', 0),
                    
                    # 音频特征
                    'valence': audio_emotion.iloc[i].get('valence', 0),
                    'energy': audio_emotion.iloc[i].get('energy', 0),
                    'danceability': audio_emotion.iloc[i].get('danceability', 0),
                    'tempo': audio_emotion.iloc[i].get('tempo', 0),
                    'loudness': audio_emotion.iloc[i].get('loudness', 0),
                    'acousticness': audio_emotion.iloc[i].get('acousticness', 0),
                    'instrumentalness': audio_emotion.iloc[i].get('instrumentalness', 0),
                    'speechiness': audio_emotion.iloc[i].get('speechiness', 0),
                    'liveness': audio_emotion.iloc[i].get('liveness', 0),
                    
                    # 标签
                    'emotion_label': emotion
                }
                paired_data.append(paired_row)
        
        self.balanced_data = pd.DataFrame(paired_data)
        
        print(f"\n生成平衡数据集: {len(self.balanced_data)} 个样本")
        print("平衡后分布:")
        balanced_dist = self.balanced_data['emotion_label'].value_counts()
        for emotion, count in balanced_dist.items():
            print(f"  {emotion}: {count}")
        
        return self.balanced_data
    
    def _augment_samples(self, df, target_count):
        """增强极少样本"""
        augmented = []
        
        # 原始样本
        for _, row in df.iterrows():
            augmented.append(row)
        
        # 生成变体
        while len(augmented) < target_count:
            base_sample = df.sample(n=1).iloc[0]
            # 添加噪声创建新样本
            new_sample = base_sample.copy()
            
            # 对数值特征添加小的随机噪声
            numeric_cols = ['sentiment_polarity', 'emotion_intensity', 'positive_ratio', 'negative_ratio']
            for col in numeric_cols:
                if col in new_sample.index:
                    new_sample[col] += np.random.normal(0, 0.05)
            
            augmented.append(new_sample)
        
        return pd.DataFrame(augmented[:target_count])
    
    def prepare_features(self):
        """准备特征"""
        print("\n3. 特征准备")
        print("-"*50)
        
        if self.balanced_data is None:
            raise ValueError("请先创建平衡数据集")
        
        # 文本特征
        text_features = ['text_length', 'sentiment_polarity', 'emotion_intensity',
                        'positive_ratio', 'negative_ratio']
        available_text = [f for f in text_features if f in self.balanced_data.columns]
        X_text = self.balanced_data[available_text].fillna(0).values
        X_text = self.scaler_text.fit_transform(X_text)
        
        # 音频特征
        audio_features = ['valence', 'energy', 'danceability', 'tempo', 
                         'loudness', 'acousticness', 'instrumentalness',
                         'speechiness', 'liveness']
        available_audio = [f for f in audio_features if f in self.balanced_data.columns]
        X_audio = self.balanced_data[available_audio].fillna(0).values
        X_audio = self.scaler_audio.fit_transform(X_audio)
        
        # 合并特征
        X = np.hstack([X_text, X_audio])
        
        # 标签
        y = self.label_encoder.fit_transform(self.balanced_data['emotion_label'])
        
        print(f"特征维度: {X.shape}")
        print(f"文本特征: {len(available_text)} 个")
        print(f"音频特征: {len(available_audio)} 个")
        print(f"总特征: {X.shape[1]} 个")
        print(f"类别: {list(self.label_encoder.classes_)}")
        
        return X, y
    
    def train_models(self, X, y):
        """训练多个模型"""
        print("\n4. 模型训练")
        print("-"*50)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 模型定义
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',  # 自动平衡权重
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        }
        
        # 训练和评估
        results = {}
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            # 训练
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1': f1,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            print(f"  准确率: {accuracy:.3f}")
            print(f"  F1分数: {f1:.3f}")
        
        self.models = results
        return results, X_test, y_test
    
    def evaluate_best_model(self):
        """评估最佳模型"""
        print("\n5. 模型评估")
        print("-"*50)
        
        # 找出最佳模型
        best_model_name = max(self.models.keys(), 
                             key=lambda k: self.models[k]['accuracy'])
        best_result = self.models[best_model_name]
        
        print(f"最佳模型: {best_model_name}")
        print(f"准确率: {best_result['accuracy']:.3f}")
        print(f"F1分数: {best_result['f1']:.3f}")
        
        # 详细分类报告
        print("\n分类报告:")
        print(classification_report(
            best_result['y_test'],
            best_result['y_pred'],
            target_names=self.label_encoder.classes_
        ))
        
        return best_result
    
    def visualize_results(self):
        """可视化结果"""
        print("\n6. 结果可视化")
        print("-"*50)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 原始数据分布
        ax1 = axes[0, 0]
        lyrics_dist = self.lyrics_df['emotion_label'].value_counts()
        ax1.bar(lyrics_dist.index, lyrics_dist.values, color='coral')
        ax1.set_title('原始歌词数据分布')
        ax1.set_ylabel('数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 平衡后数据分布
        ax2 = axes[0, 1]
        balanced_dist = self.balanced_data['emotion_label'].value_counts()
        ax2.bar(balanced_dist.index, balanced_dist.values, color='skyblue')
        ax2.set_title('平衡后数据分布')
        ax2.set_ylabel('数量')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 模型性能对比
        ax3 = axes[0, 2]
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        bars = ax3.bar(model_names, accuracies, color=colors)
        ax3.set_title('模型性能对比')
        ax3.set_ylabel('准确率')
        ax3.set_ylim([0, 1])
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 4. 混淆矩阵
        ax4 = axes[1, 0]
        best_model_name = max(self.models.keys(), 
                             key=lambda k: self.models[k]['accuracy'])
        best_result = self.models[best_model_name]
        
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=ax4)
        ax4.set_title(f'混淆矩阵 ({best_model_name})')
        ax4.set_xlabel('预测')
        ax4.set_ylabel('真实')
        
        # 5. F1分数对比
        ax5 = axes[1, 1]
        f1_scores = [self.models[name]['f1'] for name in model_names]
        bars = ax5.bar(model_names, f1_scores, color=colors)
        ax5.set_title('F1分数对比')
        ax5.set_ylabel('F1分数')
        ax5.set_ylim([0, 1])
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # 6. 特征重要性（如果是随机森林）
        ax6 = axes[1, 2]
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-10:]
            
            ax6.barh(range(len(indices)), importances[indices])
            ax6.set_yticks(range(len(indices)))
            ax6.set_yticklabels([f'特征{i}' for i in indices])
            ax6.set_title('特征重要性 (Top 10)')
            ax6.set_xlabel('重要性')
        
        plt.tight_layout()
        plt.savefig('balanced_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("图表已保存至 balanced_results.png")

def main():
    """主函数"""
    print("开始平衡的多模态音乐情感分析实验")
    print("="*70)
    
    # 初始化分析器
    analyzer = BalancedMultiModalAnalyzer()
    
    # 1. 加载和分析数据
    analyzer.load_and_analyze_data(
        'processed_chinese_lyrics_full.csv',
        'music_emotion_dataset.csv'
    )
    
    # 2. 创建平衡数据集
    balanced_data = analyzer.create_balanced_pairs(strategy='smart_pairing')
    
    # 3. 准备特征
    X, y = analyzer.prepare_features()
    
    # 4. 训练模型
    results, X_test, y_test = analyzer.train_models(X, y)
    
    # 5. 评估最佳模型
    best_result = analyzer.evaluate_best_model()
    
    # 6. 可视化
    analyzer.visualize_results()
    
    # 7. 保存结果
    balanced_data.to_csv('balanced_multimodal_dataset.csv', index=False, encoding='utf-8-sig')
    print("\n平衡数据集已保存至 balanced_multimodal_dataset.csv")
    
    # 8. 总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    print(f"原始数据: 歌词不平衡(happy仅16个) + 音频平衡(每类500个)")
    print(f"处理策略: 智能配对 + 过采样/欠采样")
    print(f"最终数据: 每类约400个样本，共{len(balanced_data)}个")
    print(f"最佳模型: {max(results.keys(), key=lambda k: results[k]['accuracy'])}")
    print(f"最高准确率: {max(r['accuracy'] for r in results.values()):.3f}")
    print("="*70)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
    
'''
1. 数据加载与分析
--------------------------------------------------
歌词数据集: 2479 样本
音频数据集: 2000 样本

原始数据分布:
歌词数据集:
  calm: 1721 (69.4%)
  sad: 572 (23.1%)
  energetic: 170 (6.9%)
  happy: 16 (0.6%)

音频数据集:
  calm: 500 (25.0%)
  energetic: 500 (25.0%)
  sad: 500 (25.0%)
  happy: 500 (25.0%)

2. 创建平衡数据集
--------------------------------------------------
使用智能配对策略...
  happy: 16 个样本，使用增强过采样
  sad: 572 个样本，使用欠采样
  calm: 1721 个样本，使用欠采样
  energetic: 170 个样本，使用随机过采样

生成平衡数据集: 1600 个样本
平衡后分布:
  happy: 400
  sad: 400
  calm: 400
  energetic: 400

3. 特征准备
--------------------------------------------------
特征维度: (1600, 14)
文本特征: 5 个
音频特征: 9 个
总特征: 14 个
类别: ['calm', 'energetic', 'happy', 'sad']

4. 模型训练
--------------------------------------------------
训练集: 1280 样本
测试集: 320 样本

训练 Random Forest...
  准确率: 0.984
  F1分数: 0.984

训练 Gradient Boosting...
  准确率: 0.997
  F1分数: 0.997

训练 Logistic Regression...
  准确率: 0.975
  F1分数: 0.975

5. 模型评估
--------------------------------------------------
最佳模型: Gradient Boosting
准确率: 0.997
F1分数: 0.997

分类报告:
              precision    recall  f1-score   support

        calm       1.00      1.00      1.00        80
   energetic       1.00      0.99      0.99        80
       happy       1.00      1.00      1.00        80
         sad       0.99      1.00      0.99        80

    accuracy                           1.00       320
   macro avg       1.00      1.00      1.00       320
weighted avg       1.00      1.00      1.00       320


6. 结果可视化
--------------------------------------------------

图表已保存至 balanced_results.png

平衡数据集已保存至 balanced_multimodal_dataset.csv

======================================================================
实验总结
======================================================================
原始数据: 歌词不平衡(happy仅16个) + 音频平衡(每类500个)
处理策略: 智能配对 + 过采样/欠采样
最终数据: 每类约400个样本，共1600个
最佳模型: Gradient Boosting
最高准确率: 0.997
======================================================================

'''