# 深度学习 BERT + CNN-LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print("="*70)

# ======================== 1. 数据集类定义 ========================
class MusicEmotionDataset(Dataset):
    """
    多模态音乐情感数据集
    
    功能：
    - 处理中文歌词文本
    - 处理音频特征
    - 返回BERT输入和音频特征张量
    """
    
    def __init__(self, texts, audio_features, labels, tokenizer, max_length=512):
        self.texts = texts
        self.audio_features = audio_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 处理文本
        text = str(self.texts[idx])
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 处理音频特征
        audio = torch.tensor(self.audio_features[idx], dtype=torch.float32)
        
        # 处理标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'audio_features': audio,
            'labels': label
        }

# ======================== 2. 模型架构定义 ========================

class TextEncoder(nn.Module):
    """
    文本编码器 - 基于BERT
    
    架构说明：
    1. 使用预训练的BERT模型提取文本特征
    2. 支持微调和特征提取两种模式
    3. 输出768维的文本表示
    """
    
    def __init__(self, bert_model_name='bert-base-chinese', freeze_bert=False):
        super(TextEncoder, self).__init__()
        
        # 加载预训练BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 是否冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 额外的投影层
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS]的输出作为句子表示
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        
        # 投影到共享空间
        text_features = self.projection(pooled_output)  # [batch_size, 256]
        
        return text_features, outputs.last_hidden_state

class AudioEncoder(nn.Module):
    """
    音频编码器 - CNN + LSTM
    
    架构说明：
    1. CNN层：提取局部音频模式
    2. LSTM层：捕捉时序依赖
    3. 注意力机制：聚合重要特征
    """
    
    def __init__(self, input_dim, hidden_dim=256):
        super(AudioEncoder, self).__init__()
        
        # 1D CNN层
        self.cnn = nn.Sequential(
            # 第一层CNN
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第二层CNN
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # 第三层CNN
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 特征投影
        self.feature_projection = nn.Linear(input_dim, 128)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim * 2, 256)
        
    def forward(self, audio_features):
        batch_size = audio_features.size(0)
        
        # CNN处理
        # 将音频特征reshape为CNN输入格式
        x = audio_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        cnn_out = self.cnn(x).squeeze(-1)  # [batch_size, 256]
        
        # 特征投影用于LSTM
        projected = self.feature_projection(audio_features)  # [batch_size, 128]
        projected = projected.unsqueeze(1)  # [batch_size, 1, 128]
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(projected)  # [batch_size, 1, 512]
        
        # 自注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 合并CNN和LSTM特征
        combined = torch.cat([
            cnn_out,
            attn_out.squeeze(1)
        ], dim=-1)[:, :512]  # 确保维度一致
        
        # 最终投影
        audio_features = self.output_projection(combined[:, :512])
        
        return audio_features

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    
    功能：
    1. 计算文本和音频特征之间的注意力权重
    2. 实现双向注意力融合
    3. 生成增强的跨模态表示
    """
    
    def __init__(self, hidden_dim=256):
        super(CrossModalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 查询、键、值投影
        self.W_q_text = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_text = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_text = nn.Linear(hidden_dim, hidden_dim)
        
        self.W_q_audio = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_audio = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_audio = nn.Linear(hidden_dim, hidden_dim)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, text_features, audio_features):
        # 文本作为查询，音频作为键值
        text_query = self.W_q_text(text_features).unsqueeze(0)
        audio_key = self.W_k_audio(audio_features).unsqueeze(0)
        audio_value = self.W_v_audio(audio_features).unsqueeze(0)
        
        # 计算文本到音频的注意力
        text_to_audio, _ = self.multihead_attn(text_query, audio_key, audio_value)
        text_to_audio = text_to_audio.squeeze(0)
        
        # 音频作为查询，文本作为键值
        audio_query = self.W_q_audio(audio_features).unsqueeze(0)
        text_key = self.W_k_text(text_features).unsqueeze(0)
        text_value = self.W_v_text(text_features).unsqueeze(0)
        
        # 计算音频到文本的注意力
        audio_to_text, _ = self.multihead_attn(audio_query, text_key, text_value)
        audio_to_text = audio_to_text.squeeze(0)
        
        # 双向融合
        fused_features = self.output_layer(
            torch.cat([text_to_audio, audio_to_text], dim=-1)
        )
        
        return fused_features

class MultiModalMusicEmotionModel(nn.Module):
    """
    完整的多模态音乐情感分析模型
    
    架构组成：
    1. 文本编码器（BERT）
    2. 音频编码器（CNN-LSTM）
    3. 跨模态注意力
    4. 融合层
    5. 分类器
    """
    
    def __init__(self, num_classes, audio_input_dim, 
                 bert_model_name='bert-base-chinese',
                 freeze_bert=False, dropout=0.3):
        super(MultiModalMusicEmotionModel, self).__init__()
        
        # 编码器
        self.text_encoder = TextEncoder(bert_model_name, freeze_bert)
        self.audio_encoder = AudioEncoder(audio_input_dim)
        
        # 跨模态注意力
        self.cross_attention = CrossModalAttention(hidden_dim=256)
        
        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Sigmoid()
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # 辅助任务：情感强度回归
        self.intensity_regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1之间的强度值
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids, audio_features):
        # 1. 编码
        text_features, text_hidden = self.text_encoder(
            input_ids, attention_mask, token_type_ids
        )
        audio_features = self.audio_encoder(audio_features)
        
        # 2. 跨模态注意力
        cross_features = self.cross_attention(text_features, audio_features)
        
        # 3. 门控融合
        gate_weights = self.gate(torch.cat([text_features, audio_features], dim=-1))
        gated_features = gate_weights * text_features + (1 - gate_weights) * audio_features
        
        # 4. 特征融合
        combined_features = torch.cat([
            text_features,
            audio_features,
            cross_features
        ], dim=-1)
        
        fused_features = self.fusion(combined_features)
        
        # 5. 输出
        emotion_logits = self.classifier(fused_features)
        intensity = self.intensity_regressor(fused_features)
        
        return emotion_logits, intensity, fused_features

# ======================== 3. 训练和评估函数 ========================

class DeepLearningTrainer:
    """
    深度学习模型训练器
    """
    
    def __init__(self, model, device, learning_rate=2e-5):
        self.model = model.to(device)
        self.device = device
        
        # 优化器 - 使用不同的学习率
        bert_params = list(model.text_encoder.bert.parameters())
        other_params = [p for n, p in model.named_parameters() 
                       if not n.startswith('text_encoder.bert')]
        
        self.optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': learning_rate * 0.1},  # BERT用更小的学习率
            {'params': other_params, 'lr': learning_rate}
        ])
        
        # 损失函数
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-6
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            # 准备数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            emotion_logits, intensity, _ = self.model(
                input_ids, attention_mask, token_type_ids, audio_features
            )
            
            # 计算损失
            cls_loss = self.criterion_cls(emotion_logits, labels)
            # 这里可以添加强度回归损失（如果有标签）
            total_batch_loss = cls_loss
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            _, predicted = torch.max(emotion_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': total_batch_loss.item(),
                'acc': correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                emotion_logits, intensity, _ = self.model(
                    input_ids, attention_mask, token_type_ids, audio_features
                )
                
                loss = self.criterion_cls(emotion_logits, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(emotion_logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, epochs=10):
        """完整训练流程"""
        print("\n开始训练深度学习模型...")
        print("="*50)
        
        best_accuracy = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 学习率调度
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"保存最佳模型，准确率: {best_accuracy:.4f}")
        
        return best_accuracy

# ======================== 4. 完整实验流程 ========================

class DeepLearningExperiment:
    """
    深度学习实验管理器
    """
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, lyrics_path, audio_path):
        """准备数据"""
        print("加载数据集...")
        
        # 加载数据
        lyrics_df = pd.read_csv(lyrics_path)
        audio_df = pd.read_csv(audio_path)
        
        # 简单的数据配对（实际应用中需要更复杂的匹配逻辑）
        n_samples = min(len(lyrics_df), len(audio_df))
        
        # 提取文本
        texts = lyrics_df['processed_clean_text'].fillna('').values[:n_samples]
        
        # 提取音频特征
        audio_cols = ['valence', 'energy', 'danceability', 'tempo', 
                     'loudness', 'acousticness', 'instrumentalness', 
                     'speechiness', 'liveness']
        audio_features = audio_df[audio_cols].fillna(0).values[:n_samples]
        
        # 提取标签
        labels = lyrics_df['emotion_label'].values[:n_samples]
        labels = self.label_encoder.fit_transform(labels)
        
        return texts, audio_features, labels
    
    def create_data_loaders(self, texts, audio_features, labels, 
                           batch_size=16, test_size=0.2):
        """创建数据加载器"""
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 数据分割
        X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = \
            train_test_split(texts, audio_features, labels, 
                           test_size=test_size, random_state=42, stratify=labels)
        
        # 创建数据集
        train_dataset = MusicEmotionDataset(
            X_text_train, X_audio_train, y_train, self.tokenizer
        )
        test_dataset = MusicEmotionDataset(
            X_text_test, X_audio_test, y_test, self.tokenizer
        )
        
        # 创建加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def run_experiment(self, lyrics_path, audio_path, epochs=10):
        """运行完整实验"""
        print("="*70)
        print("深度学习多模态音乐情感分析实验")
        print("="*70)
        
        # 准备数据
        texts, audio_features, labels = self.prepare_data(lyrics_path, audio_path)
        print(f"数据集大小: {len(texts)} 样本")
        print(f"情感类别: {len(np.unique(labels))} 类")
        
        # 创建数据加载器
        train_loader, test_loader = self.create_data_loaders(
            texts, audio_features, labels
        )
        
        # 初始化模型
        num_classes = len(self.label_encoder.classes_)
        audio_input_dim = audio_features.shape[1]
        
        self.model = MultiModalMusicEmotionModel(
            num_classes=num_classes,
            audio_input_dim=audio_input_dim,
            freeze_bert=False  # 微调BERT
        )
        
        # 初始化训练器
        self.trainer = DeepLearningTrainer(self.model, device)
        
        # 训练模型
        best_accuracy = self.trainer.train(train_loader, test_loader, epochs)
        
        # 最终评估
        print("\n最终评估...")
        _, test_acc, predictions, true_labels = self.trainer.evaluate(test_loader)
        
        print(f"\n最终测试准确率: {test_acc:.4f}")
        
        # 生成分类报告
        print("\n分类报告:")
        print(classification_report(
            true_labels, predictions,
            target_names=self.label_encoder.classes_
        ))
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return self.model, test_acc
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.trainer.train_losses, label='Train Loss')
        axes[0].plot(self.trainer.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('训练和验证损失')
        axes[0].legend()
        axes[0].grid(True)
        
        # 准确率曲线
        axes[1].plot(self.trainer.val_accuracies, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('验证准确率')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

# ======================== 5. 模型解释和可视化 ========================

class ModelInterpreter:
    """
    模型解释器 - 可视化注意力权重和特征
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def visualize_attention(self, text, audio_features):
        """可视化注意力权重"""
        self.model.eval()
        
        # 准备输入
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取中间特征
            text_features, text_hidden = self.model.text_encoder(
                input_ids, attention_mask, token_type_ids
            )
            audio_features = self.model.audio_encoder(audio_tensor)
            
            # 这里可以提取和可视化注意力权重
            # 具体实现取决于模型架构
        
        print("注意力可视化完成")
        
    def extract_features(self, data_loader):
        """提取学习到的特征用于可视化"""
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                labels = batch['labels']
                
                _, _, fused_features = self.model(
                    input_ids, attention_mask, token_type_ids, audio_features
                )
                
                all_features.append(fused_features.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.array(all_labels)
        
        return features, labels

# ======================== 主函数 ========================

def main():
    """主函数 - 运行完整的深度学习实验"""
    
    # 检查BERT模型是否可用
    try:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        print("BERT模型加载成功")
    except:
        print("警告: 无法加载BERT模型，请安装transformers库")
        print("pip install transformers")
        return
    
    # 初始化实验
    experiment = DeepLearningExperiment()
    
    # 运行实验
    model, accuracy = experiment.run_experiment(
        lyrics_path='processed_chinese_lyrics_full.csv',
        audio_path='music_emotion_dataset.csv',
        epochs=10
    )
    
    print("\n" + "="*70)
    print("实验完成!")
    print(f"最终准确率: {accuracy:.4f}")
    print("="*70)
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main()

'''
使用设备: cpu
======================================================================
BERT模型加载成功

加载数据集...
数据集大小: 2000 样本
情感类别: 4 类

开始训练深度学习模型...
==================================================

Epoch 1/10
Training:   0%|                                                                                | 0/100 [00:00<?, ?it/s]
'''