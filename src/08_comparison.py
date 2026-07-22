import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

print("快速多模态音乐情感分析系统")
print("="*70)

# ======================== 1：轻量级深度学习 ========================

class LightweightModel(nn.Module):
    """轻量级深度学习模型 """
    
    def __init__(self, text_dim=100, audio_dim=9, num_classes=4):
        super(LightweightModel, self).__init__()
        
        # 简单的文本处理
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # 简单的音频处理
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 融合和分类
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, text_features, audio_features):
        text_out = self.text_fc(text_features)
        audio_out = self.audio_fc(audio_features)
        combined = torch.cat([text_out, audio_out], dim=1)
        return self.classifier(combined)

def train_lightweight_model(X_text, X_audio, y, epochs=5):
    """训练轻量级模型"""
    print("\n训练轻量级深度学习模型...")
    
    # 数据分割
    X_text_train, X_text_val, X_audio_train, X_audio_val, y_train, y_val = \
        train_test_split(X_text, X_audio, y, test_size=0.2, random_state=42)
    
    # 转换为张量
    X_text_train = torch.FloatTensor(X_text_train)
    X_text_val = torch.FloatTensor(X_text_val)
    X_audio_train = torch.FloatTensor(X_audio_train)
    X_audio_val = torch.FloatTensor(X_audio_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    # 创建模型
    model = LightweightModel(text_dim=X_text_train.shape[1], 
                            audio_dim=X_audio_train.shape[1])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_text_train, X_audio_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_text_val, X_audio_val)
            _, predicted = torch.max(val_outputs, 1)
            accuracy = (predicted == y_val).float().mean()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Val Acc: {accuracy:.4f}, 时间: {epoch_time:.2f}秒")
    
    return model, accuracy.item()

# ======================== 2：传统机器学习集成 ========================

class EnsembleApproach:
    """高效的传统机器学习集成方法"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
            'lr': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.weights = None
        
    def train(self, X_train, y_train, X_val, y_val):
        """训练所有模型"""
        print("\n训练集成模型...")
        
        results = {}
        for name, model in self.models.items():
            start_time = time.time()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            train_time = time.time() - start_time
            results[name] = accuracy
            
            print(f"{name.upper()}: 准确率={accuracy:.4f}, 训练时间={train_time:.2f}秒")
        
        # 设置权重
        total_acc = sum(results.values())
        self.weights = {name: acc/total_acc for name, acc in results.items()}
        
        return results
    
    def predict(self, X):
        """加权预测"""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            predictions.append(pred * self.weights[name])
        
        final_pred = np.sum(predictions, axis=0)
        return np.argmax(final_pred, axis=1)

# ======================== 数据处理 ========================

def prepare_data(lyrics_path, audio_path, n_samples=None):
    """快速数据准备"""
    print("准备数据...")
    
    # 加载数据
    lyrics_df = pd.read_csv(lyrics_path, nrows=n_samples)
    audio_df = pd.read_csv(audio_path, nrows=n_samples)
    
    # 文本特征 - 使用简单的统计特征
    text_features = []
    for col in lyrics_df.columns:
        if any(keyword in col.lower() for keyword in ['length', 'count', 'ratio', 'score', 'level']):
            if col in lyrics_df.columns and lyrics_df[col].dtype in [np.float64, np.int64]:
                text_features.append(col)
    
    if text_features:
        X_text = lyrics_df[text_features].fillna(0).values
    else:
        # 生成随机特征
        X_text = np.random.randn(n_samples, 100)
    
    # 音频特征
    audio_cols = ['valence', 'energy', 'danceability', 'tempo', 
                 'loudness', 'acousticness', 'instrumentalness', 
                 'speechiness', 'liveness']
    
    available_audio = [col for col in audio_cols if col in audio_df.columns]
    if available_audio:
        X_audio = audio_df[available_audio].fillna(0).values
    else:
        X_audio = np.random.randn(n_samples, 9)
    
    # 标签
    label_col = None
    for col in ['emotion_label', 'emotion', 'dominant_emotion']:
        if col in lyrics_df.columns:
            label_col = col
            break
        if col in audio_df.columns:
            label_col = col
            break
    
    if label_col:
        if label_col in lyrics_df.columns:
            labels = lyrics_df[label_col].values
        else:
            labels = audio_df[label_col].values
    else:
        labels = np.random.choice(['happy', 'sad', 'calm', 'energetic'], n_samples)
    
    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # 标准化
    scaler = StandardScaler()
    X_text = scaler.fit_transform(X_text)
    X_audio = scaler.fit_transform(X_audio)
    
    print(f"文本特征: {X_text.shape}")
    print(f"音频特征: {X_audio.shape}")
    print(f"标签类别: {list(le.classes_)}")
    
    return X_text, X_audio, y, le

# ======================== 主函数 ========================

def main():
    """主函数"""
    start_time = time.time()
    
    # 准备数据
    X_text, X_audio, y, le = prepare_data(
        'processed_chinese_lyrics_full.csv',
        'music_emotion_dataset.csv',
        n_samples=2000  # 使用较少样本以加快速度
    )
    # 合并特征
    X_combined = np.hstack([X_text, X_audio])
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    # 方案1：轻量级深度学习
    print("\n" + "="*50)
    print("1: 轻量级深度学习")
    print("="*50)
    
    try:
        X_text_train, X_text_test = X_train[:, :X_text.shape[1]], X_test[:, :X_text.shape[1]]
        X_audio_train, X_audio_test = X_train[:, X_text.shape[1]:], X_test[:, X_text.shape[1]:]
        
        model, accuracy = train_lightweight_model(
            X_text_train, X_audio_train, y_train, epochs=10
        )
        results['深度学习'] = accuracy
        
        # 测试集评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(
                torch.FloatTensor(X_text_test),
                torch.FloatTensor(X_audio_test)
            )
            _, predicted = torch.max(test_outputs, 1)
            test_accuracy = accuracy_score(y_test, predicted.numpy())
            results['深度学习_测试'] = test_accuracy
            
    except Exception as e:
        print(f"深度学习训练失败: {e}")
    
    # 2：传统机器学习模型集成训练
    print("\n" + "="*50)
    print("2: 传统机器学习集成")
    print("="*50)
    
    ensemble = EnsembleApproach()
    ensemble_results = ensemble.train(X_train, y_train, X_test, y_test)
    
    # 集成预测
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    results['集成模型'] = ensemble_accuracy
    
    # 总结
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("实验结果总结")
    print("="*70)
    
    for method, acc in results.items():
        print(f"{method}: {acc:.4f}")
    
    print(f"\n总运行时间: {total_time:.2f}秒")
    
    # 分类报告
    print("\n最佳模型分类报告:")
    if 'deep_predicted' in locals():
        print(classification_report(y_test, predicted.numpy(), 
                                   target_names=le.classes_))
    else:
        print(classification_report(y_test, y_pred_ensemble, 
                                   target_names=le.classes_))
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main()

'''
准备数据...
文本特征: (2000, 14)
音频特征: (2000, 9)
标签类别: ['calm', 'energetic', 'happy', 'sad']

==================================================
1: 轻量级深度学习
==================================================

训练轻量级深度学习模型...
模型参数数量: 6628
Epoch 1/15, Loss: 1.4948, Val Acc: 0.0156, 时间: 0.02秒
Epoch 2/15, Loss: 1.4774, Val Acc: 0.0188, 时间: 0.02秒
Epoch 3/15, Loss: 1.4574, Val Acc: 0.0281, 时间: 0.01秒
Epoch 4/15, Loss: 1.4397, Val Acc: 0.0500, 时间: 0.01秒
Epoch 5/15, Loss: 1.4246, Val Acc: 0.0719, 时间: 0.01秒
Epoch 6/15, Loss: 1.4087, Val Acc: 0.1125, 时间: 0.01秒
Epoch 7/15, Loss: 1.3906, Val Acc: 0.1813, 时间: 0.01秒
Epoch 8/15, Loss: 1.3732, Val Acc: 0.2531, 时间: 0.01秒
Epoch 9/15, Loss: 1.3575, Val Acc: 0.3219, 时间: 0.02秒
Epoch 10/15, Loss: 1.3422, Val Acc: 0.3969, 时间: 0.02秒
Epoch 11/15, Loss: 1.3222, Val Acc: 0.5219, 时间: 0.01秒
Epoch 12/15, Loss: 1.3078, Val Acc: 0.5969, 时间: 0.01秒
Epoch 13/15, Loss: 1.2946, Val Acc: 0.6625, 时间: 0.01秒
Epoch 14/15, Loss: 1.2671, Val Acc: 0.6969, 时间: 0.01秒
Epoch 15/15, Loss: 1.2488, Val Acc: 0.6906, 时间: 0.02秒

==================================================
方案2: 传统机器学习集成
==================================================

训练集成模型...
RF: 准确率=0.9600, 训练时间=0.56秒
GB: 准确率=0.9700, 训练时间=6.02秒
LR: 准确率=0.9675, 训练时间=0.03秒

======================================================================
实验结果总结
======================================================================
深度学习: 0.6906
深度学习_测试: 0.6900
集成模型: 0.9775

总运行时间: 7.03秒

最佳模型分类报告:
              precision    recall  f1-score   support

        calm       0.98      0.99      0.98       275
   energetic       1.00      0.96      0.98        27
       happy       0.00      0.00      0.00         3
         sad       0.97      0.98      0.97        95

    accuracy                           0.98       400
   macro avg       0.74      0.73      0.73       400
weighted avg       0.97      0.98      0.97       400


======================================================================
实验完成！

'''