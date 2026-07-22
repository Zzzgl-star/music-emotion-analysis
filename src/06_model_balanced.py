# BERT + CNN-LSTM 多模态音乐情感分析 - 平衡数据版
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# 检查设备和BERT可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 尝试加载BERT
USE_BERT = False
BERT_AVAILABLE = False
try:
    from transformers import BertTokenizer, BertModel
    tokenizer_test = BertTokenizer.from_pretrained('bert-base-chinese')
    BERT_AVAILABLE = True
    print("BERT模型可用")
except Exception as e:
    print(f"BERT不可用: {e}")
    print("将使用备用方案")

print("="*70)

# ======================== 数据平衡模块 ========================
class DataBalancer:
    """数据平衡处理器"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_and_balance(self, lyrics_path, audio_path):
        """加载并平衡数据"""
        print("1. 加载数据...")
        
        # 加载数据
        lyrics_df = pd.read_csv(lyrics_path)
        audio_df = pd.read_csv(audio_path)
        
        print(f"原始数据分布:")
        print(f"歌词: {lyrics_df['emotion_label'].value_counts().to_dict()}")
        print(f"音频: {audio_df['emotion_label'].value_counts().to_dict()}")
        
        # 创建平衡配对
        print("\n2. 创建平衡数据集...")
        balanced_data = self._create_balanced_pairs(lyrics_df, audio_df)
        
        return balanced_data
    
    def _create_balanced_pairs(self, lyrics_df, audio_df):
        """创建平衡的配对数据"""
        emotions = ['happy', 'sad', 'calm', 'energetic']
        target_per_class = 400
        paired_data = []
        
        for emotion in emotions:
            # 音频数据（平衡的）
            audio_emotion = audio_df[audio_df['emotion_label'] == emotion]
            if len(audio_emotion) > target_per_class:
                audio_emotion = audio_emotion.sample(n=target_per_class, random_state=42)
            
            # 歌词数据（不平衡的）
            lyrics_emotion = lyrics_df[lyrics_df['emotion_label'] == emotion]
            
            if len(lyrics_emotion) < target_per_class:
                # 过采样
                if len(lyrics_emotion) < 20:
                    # Happy类特殊处理
                    lyrics_sampled = self._augment_few_samples(lyrics_emotion, target_per_class)
                else:
                    lyrics_sampled = lyrics_emotion.sample(n=target_per_class, replace=True, random_state=42)
            else:
                # 欠采样
                lyrics_sampled = lyrics_emotion.sample(n=target_per_class, replace=False, random_state=42)
            
            # 重置索引
            audio_emotion = audio_emotion.reset_index(drop=True)
            lyrics_sampled = lyrics_sampled.reset_index(drop=True)
            
            # 配对
            for i in range(min(len(audio_emotion), len(lyrics_sampled))):
                paired_data.append({
                    'text': lyrics_sampled.iloc[i].get('processed_clean_text', ''),
                    'valence': audio_emotion.iloc[i].get('valence', 0.5),
                    'energy': audio_emotion.iloc[i].get('energy', 0.5),
                    'danceability': audio_emotion.iloc[i].get('danceability', 0.5),
                    'tempo': audio_emotion.iloc[i].get('tempo', 120),
                    'loudness': audio_emotion.iloc[i].get('loudness', -10),
                    'acousticness': audio_emotion.iloc[i].get('acousticness', 0.5),
                    'instrumentalness': audio_emotion.iloc[i].get('instrumentalness', 0.5),
                    'speechiness': audio_emotion.iloc[i].get('speechiness', 0.5),
                    'liveness': audio_emotion.iloc[i].get('liveness', 0.5),
                    'emotion': emotion
                })
        
        balanced_df = pd.DataFrame(paired_data)
        print(f"平衡后分布: {balanced_df['emotion'].value_counts().to_dict()}")
        
        return balanced_df
    
    def _augment_few_samples(self, df, target_count):
        """增强极少样本（如Happy类）"""
        augmented = []
        
        # 原始样本
        for _, row in df.iterrows():
            augmented.append(row.to_dict())
        
        # 生成变体
        while len(augmented) < target_count:
            base = df.sample(n=1).iloc[0].to_dict()
            # 可以在这里添加文本增强逻辑
            augmented.append(base)
        
        return pd.DataFrame(augmented[:target_count])

# ======================== 模型定义 ========================

class TextEncoder(nn.Module):
    """文本编码器 - BERT或LSTM"""
    
    def __init__(self, vocab_size=5000, use_bert=False):
        super(TextEncoder, self).__init__()
        self.use_bert = use_bert
        
        if use_bert and BERT_AVAILABLE:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained('bert-base-chinese')
            # 冻结BERT以加快训练
            for param in self.bert.parameters():
                param.requires_grad = False
            self.projection = nn.Linear(768, 256)
        else:
            # 备用LSTM方案
            self.embedding = nn.Embedding(vocab_size, 128)
            self.lstm = nn.LSTM(128, 256, num_layers=2, 
                               batch_first=True, bidirectional=True, dropout=0.2)
            self.projection = nn.Linear(512, 256)
    
    def forward(self, input_ids, attention_mask=None):
        if self.use_bert and BERT_AVAILABLE:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.pooler_output
            return self.projection(pooled)
        else:
            embedded = self.embedding(input_ids)
            lstm_out, (hidden, _) = self.lstm(embedded)
            # 使用最后时刻的输出
            last_output = lstm_out[:, -1, :]
            return self.projection(last_output)

class AudioCNNLSTM(nn.Module):
    """音频编码器 - CNN + LSTM"""
    
    def __init__(self, input_dim=9):
        super(AudioCNNLSTM, self).__init__()
        
        # 1D CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveMaxPool1d(4)
        )
        
        # LSTM
        self.lstm = nn.LSTM(128*4, 256, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.2)
        
        # 投影层
        self.projection = nn.Linear(512, 256)
        
    def forward(self, x):
        # x: [batch, features]
        # 扩展维度用于CNN
        x = x.unsqueeze(1)  # [batch, 1, features]
        
        # CNN处理
        cnn_out = self.cnn(x)  # [batch, 128, 4]
        
        # 重塑用于LSTM
        batch_size = cnn_out.size(0)
        cnn_out = cnn_out.view(batch_size, 1, -1)  # [batch, 1, 512]
        
        # LSTM处理
        lstm_out, _ = self.lstm(cnn_out)  # [batch, 1, 512]
        
        # 投影
        output = self.projection(lstm_out.squeeze(1))
        
        return output

class MultiModalFusionModel(nn.Module):
    """多模态融合模型"""
    
    def __init__(self, num_classes=4, vocab_size=5000, audio_dim=9, use_bert=False):
        super(MultiModalFusionModel, self).__init__()
        
        # 编码器
        self.text_encoder = TextEncoder(vocab_size, use_bert)
        self.audio_encoder = AudioCNNLSTM(audio_dim)
        
        # 注意力融合
        self.attention = nn.MultiheadAttention(256, num_heads=8, dropout=0.1)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, text_input, audio_input, attention_mask=None):
        # 编码
        text_features = self.text_encoder(text_input, attention_mask)
        audio_features = self.audio_encoder(audio_input)
        
        # 注意力融合
        text_features_exp = text_features.unsqueeze(0)
        audio_features_exp = audio_features.unsqueeze(0)
        
        attended, _ = self.attention(text_features_exp, 
                                    audio_features_exp, 
                                    audio_features_exp)
        attended = attended.squeeze(0)
        
        # 门控融合
        combined = torch.cat([text_features, audio_features], dim=1)
        gate_weights = self.gate(combined)
        fused = gate_weights * text_features + (1 - gate_weights) * audio_features
        
        # 分类
        output = self.classifier(fused)
        
        return output

# ======================== 数据集类 ========================

class MusicDataset(Dataset):
    """音乐数据集"""
    
    def __init__(self, texts, audio_features, labels, tokenizer=None, max_length=256):
        self.texts = texts
        self.audio_features = audio_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 如果没有tokenizer，创建字符级词汇表
        if tokenizer is None:
            self.vocab = self._build_vocab(texts)
    
    def _build_vocab(self, texts):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for text in texts:
            for char in str(text):
                if char not in vocab:
                    vocab[char] = len(vocab)
        return vocab
    
    def _text_to_ids(self, text):
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()
        else:
            # 字符级编码
            ids = [self.vocab.get(char, 1) for char in str(text)[:self.max_length]]
            ids += [0] * (self.max_length - len(ids))
            mask = [1 if i > 0 else 0 for i in ids]
            return torch.tensor(ids[:self.max_length]), torch.tensor(mask[:self.max_length])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        input_ids, attention_mask = self._text_to_ids(text)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'audio_features': torch.tensor(self.audio_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ======================== 训练函数 ========================

def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, audio_features, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_loss)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_features = batch['audio_features'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, audio_features, attention_mask)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return train_losses, val_accuracies

# ======================== 主函数 ========================

def main():
    """主函数"""
    print("BERT + CNN-LSTM 多模态音乐情感分析")
    print("="*70)
    
    # 1. 数据平衡
    balancer = DataBalancer()
    balanced_data = balancer.load_and_balance(
        'processed_chinese_lyrics_full.csv',
        'music_emotion_dataset.csv'
    )
    
    # 2. 准备数据
    print("\n3. 准备训练数据...")
    
    # 提取特征
    texts = balanced_data['text'].values
    audio_features = balanced_data[['valence', 'energy', 'danceability', 
                                   'tempo', 'loudness', 'acousticness',
                                   'instrumentalness', 'speechiness', 'liveness']].values
    
    # 标准化音频特征
    scaler = StandardScaler()
    audio_features = scaler.fit_transform(audio_features)
    
    # 编码标签
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(balanced_data['emotion'])
    
    print(f"样本数: {len(texts)}")
    print(f"类别: {list(label_encoder.classes_)}")
    
    # 3. 分割数据
    X_text_train, X_text_val, X_audio_train, X_audio_val, y_train, y_val = \
        train_test_split(texts, audio_features, labels, 
                        test_size=0.2, random_state=42, stratify=labels)
    
    # 4. 创建数据集和加载器
    print("\n4. 创建数据加载器...")
    
    # 决定是否使用BERT
    if BERT_AVAILABLE and input("是否使用BERT？(y/n): ").lower() == 'y':
        USE_BERT = True
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        vocab_size = None
    else:
        USE_BERT = False
        tokenizer = None
        # 计算词汇表大小
        all_chars = set()
        for text in texts:
            all_chars.update(str(text))
        vocab_size = len(all_chars) + 2
    
    train_dataset = MusicDataset(X_text_train, X_audio_train, y_train, tokenizer)
    val_dataset = MusicDataset(X_text_val, X_audio_val, y_val, tokenizer)
    
    if not USE_BERT:
        vocab_size = len(train_dataset.vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 5. 创建模型
    print(f"\n5. 创建模型...")
    print(f"使用BERT: {USE_BERT}")
    
    model = MultiModalFusionModel(
        num_classes=len(label_encoder.classes_),
        vocab_size=vocab_size if vocab_size else 5000,
        audio_dim=9,
        use_bert=USE_BERT
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {total_params:,}")
    
    # 6. 训练
    print("\n6. 开始训练...")
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, epochs=10, device=device
    )
    
    # 7. 最终评估
    print("\n7. 最终评估...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_features = batch['audio_features'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, audio_features, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n最终准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, 
                              target_names=label_encoder.classes_))
    
    # 8. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses)
    axes[0].set_title('训练损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(val_accuracies)
    axes[1].set_title('验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('bert_cnn_lstm_results.png')
    print("\n结果已保存至 bert_cnn_lstm_results.png")
    
    # 保存模型
    torch.save(model.state_dict(), 'bert_cnn_lstm_model.pth')
    print("模型已保存至 bert_cnn_lstm_model.pth")
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"最终性能: 准确率={accuracy:.4f}, F1={f1:.4f}")
    print("="*70)
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()
