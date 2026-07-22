# 中文歌词数据集清洗与预处理
# ===============================

'''
数据集来源于和鲸社区 下载
'''

import pandas as pd
import numpy as np
import jieba
import re
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

#设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

print("中文歌词数据集清洗与预处理")
print("=" * 50)

class ChineseLyricsProcessor:
    """中文歌词数据处理器"""
    
    def __init__(self, input_file):
        self.input_file = input_file
        self.df = None
        
    def load_data(self):
        """加载数据"""
        print("\n1. 数据加载")
        print("-" * 30)
        
        self.df = pd.read_csv(self.input_file)
        
        print(f"数据加载完成")
        print(f"   总歌曲数: {len(self.df)}")
        print(f"   数据列: {list(self.df.columns)}")
        print(f"   内存使用: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # 显示数据质量概况
        print(f"\n数据质量检查:")
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                print(f"   {col}: {missing} 个缺失值 ({missing/len(self.df)*100:.1f}%)")
            else:
                print(f"   {col}: 无缺失值")
        
        return self.df
    
    def clean_lyrics_text(self):
        """清洗歌词文本"""
        print("\n2. 歌词文本清洗")
        print("-" * 30)
        
        def advanced_text_cleaning(text):
            """高级文本清洗"""
            if pd.isna(text) or text == '':
                return ''
            
            text = str(text)
            
            # 去除引号
            text = text.strip('"\'')
            
            # 统一换行符
            text = text.replace('\\n', '\n').replace('　', '\n')
            
            # 去除重复的换行符
            text = re.sub(r'\n+', '\n', text)
            
            # 去除开头结尾的空白
            text = text.strip()
            
            # 去除空行
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            return text
        
        # 清洗原始文本和clean_text
        self.df['processed_text'] = self.df['text'].apply(advanced_text_cleaning)
        self.df['processed_clean_text'] = self.df['clean_text'].apply(advanced_text_cleaning)
        
        # 统计清洗效果
        original_avg_length = self.df['text'].astype(str).str.len().mean()
        processed_avg_length = self.df['processed_text'].str.len().mean()
        
        print(f"   文本清洗完成")
        print(f"   原始平均长度: {original_avg_length:.0f} 字符")
        print(f"   清洗后平均长度: {processed_avg_length:.0f} 字符")
        
        return self.df
    
    def extract_text_features(self):
        """提取文本特征"""
        print("\n3. 文本特征提取")
        print("-" * 30)
        
        # 基础统计特征
        self.df['text_length'] = self.df['processed_clean_text'].str.len()
        self.df['line_count'] = self.df['processed_clean_text'].str.count('\n') + 1
        self.df['char_count'] = self.df['processed_clean_text'].apply(
            lambda x: len(x.replace(' ', '').replace('\n', '')) if pd.notna(x) else 0
        )
        self.df['unique_chars'] = self.df['processed_clean_text'].apply(
            lambda x: len(set(x)) if pd.notna(x) else 0
        )
        
        # 计算字符密度
        self.df['char_density'] = self.df['char_count'] / np.maximum(self.df['text_length'], 1)
        
        print(f"基础统计特征提取完成")
        
        return self.df
    
    def analyze_emotion_keywords(self):
        """分析情感关键词"""
        print("\n4. 情感关键词分析")
        print("-" * 30)
        
        # 扩展的中文情感词典
        positive_words = {
            '快乐', '开心', '高兴', '喜欢', '爱', '幸福', '甜蜜', '美好', '温暖', '阳光',
            '希望', '梦想', '微笑', '笑', '甜', '美', '好', '棒', '赞', '优秀',
            '成功', '胜利', '完美', '精彩', '动人', '感动', '兴奋', '激动', '满足', '享受'
        }
        
        negative_words = {
            '伤心', '难过', '痛苦', '悲伤', '哭', '眼泪', '孤单', '寂寞', '冷漠', '痛',
            '失望', '绝望', '忧郁', '郁闷', '沮丧', '烦恼', '担心', '害怕', '恐惧', '愤怒',
            '生气', '愤', '恨', '怨', '冷', '黑暗', '阴', '苦', '累', '疲惫'
        }
        
        energetic_words = {
            '奔跑', '飞翔', '跳舞', '狂欢', '激情', '热情', '火热', '燃烧', '爆发', '冲刺',
            '拼搏', '奋斗', '战斗', '挑战', '勇敢', '坚强', '力量', '能量', '活力', '青春'
        }
        
        calm_words = {
            '平静', '安静', '宁静', '淡然', '轻松', '舒缓', '柔和', '温柔', '慢', '缓',
            '月亮', '星空', '夜晚', '梦', '思考', '回忆', '怀念', '静静', '轻轻', '慢慢'
        }
        
        def analyze_text_emotion(text):
            """分析单个文本的情感特征"""
            if pd.isna(text) or text == '':
                return {
                    'positive_count': 0, 'negative_count': 0, 'energetic_count': 0, 'calm_count': 0,
                    'positive_ratio': 0, 'negative_ratio': 0, 'energetic_ratio': 0, 'calm_ratio': 0,
                    'sentiment_polarity': 0, 'emotion_intensity': 0, 'dominant_emotion': 'neutral'
                }
            
            # 分词
            words = list(jieba.cut(text))
            meaningful_words = [w for w in words if len(w) > 1 and w.strip()]
            
            # 统计各类情感词
            pos_count = sum(1 for word in meaningful_words if word in positive_words)
            neg_count = sum(1 for word in meaningful_words if word in negative_words)
            energetic_count = sum(1 for word in meaningful_words if word in energetic_words)
            calm_count = sum(1 for word in meaningful_words if word in calm_words)
            
            total_words = len(meaningful_words)
            
            if total_words > 0:
                pos_ratio = pos_count / total_words
                neg_ratio = neg_count / total_words
                energetic_ratio = energetic_count / total_words
                calm_ratio = calm_count / total_words
                
                sentiment_polarity = (pos_count - neg_count) / total_words
                emotion_intensity = (pos_count + neg_count + energetic_count) / total_words
                
                # 确定主导情感
                scores = {
                    'positive': pos_count + pos_ratio * 10,
                    'negative': neg_count + neg_ratio * 10,
                    'energetic': energetic_count + energetic_ratio * 10,
                    'calm': calm_count + calm_ratio * 10
                }
                dominant_emotion = max(scores.keys(), key=lambda k: scores[k]) if max(scores.values()) > 0 else 'neutral'
                
            else:
                pos_ratio = neg_ratio = energetic_ratio = calm_ratio = 0
                sentiment_polarity = emotion_intensity = 0
                dominant_emotion = 'neutral'
            
            return {
                'positive_count': pos_count,
                'negative_count': neg_count,
                'energetic_count': energetic_count,
                'calm_count': calm_count,
                'positive_ratio': pos_ratio,
                'negative_ratio': neg_ratio,
                'energetic_ratio': energetic_ratio,
                'calm_ratio': calm_ratio,
                'sentiment_polarity': sentiment_polarity,
                'emotion_intensity': emotion_intensity,
                'dominant_emotion': dominant_emotion,
                'total_meaningful_words': total_words
            }
        
        # 应用情感分析
        emotion_features = self.df['processed_clean_text'].apply(analyze_text_emotion)
        
        # 将结果转换为列
        emotion_df = pd.DataFrame(emotion_features.tolist())
        self.df = pd.concat([self.df, emotion_df], axis=1)
        
        print(f"情感关键词分析完成")
        
        # 显示情感分布
        emotion_dist = self.df['dominant_emotion'].value_counts()
        print(f"   主导情感分布:")
        for emotion, count in emotion_dist.items():
            print(f"     {emotion}: {count} 首 ({count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def process_hsk_grades(self):
        """处理HSK等级数据"""
        print("\n5. HSK等级特征处理")
        print("-" * 30)
        
        def parse_graded_data(graded_str):
            """解析graded字段"""
            if pd.isna(graded_str) or graded_str == '':
                return {}
            
            try:
                # 尝试解析为字典
                if isinstance(graded_str, str):
                    graded_dict = ast.literal_eval(graded_str)
                    return graded_dict
                else:
                    return {}
            except:
                return {}
        
        def extract_hsk_features(graded_dict):
            """从HSK数据提取特征"""
            features = {}
            
            # 计算各等级词汇数量
            for level in range(1, 7):
                words = graded_dict.get(level, [])
                features[f'hsk_level_{level}_count'] = len(words)
            
            # 计算总词汇量和各等级比例
            total_words = sum(len(graded_dict.get(i, [])) for i in range(1, 7))
            
            for level in range(1, 7):
                count = features[f'hsk_level_{level}_count']
                features[f'hsk_level_{level}_ratio'] = count / max(total_words, 1)
            
            # 计算平均HSK等级
            if total_words > 0:
                weighted_sum = sum(level * len(graded_dict.get(level, [])) for level in range(1, 7))
                features['avg_hsk_level'] = weighted_sum / total_words
                features['hsk_complexity'] = features['avg_hsk_level']  # 语言复杂度指标
            else:
                features['avg_hsk_level'] = 3.0  # 默认中等难度
                features['hsk_complexity'] = 3.0
            
            features['total_hsk_words'] = total_words
            features['unknown_chars_count'] = len(graded_dict.get(0, []))  # 未知字符数
            
            return features
        
        # 解析HSK数据
        if 'graded' in self.df.columns:
            graded_data = self.df['graded'].apply(parse_graded_data)
            hsk_features = graded_data.apply(extract_hsk_features)
            
            # 转换为DataFrame并合并
            hsk_df = pd.DataFrame(hsk_features.tolist())
            self.df = pd.concat([self.df, hsk_df], axis=1)
            
            print(f"   HSK等级特征处理完成")
            print(f"   平均HSK等级: {self.df['avg_hsk_level'].mean():.2f}")
            print(f"   语言复杂度分布:")
            
            # 将复杂度分组显示
            complexity_bins = pd.cut(self.df['avg_hsk_level'], bins=[0, 2, 3, 4, 6], labels=['简单', '中等', '复杂', '高难'])
            complexity_dist = complexity_bins.value_counts()
            for level, count in complexity_dist.items():
                print(f"     {level}: {count} 首")
        else:
            print("未找到graded字段，跳过HSK特征处理")
        
        return self.df
    
    def create_final_emotion_labels(self):
        """创建最终的情感标签"""
        print("\n6. 情感标签生成")
        print("-" * 30)
        
        def determine_final_emotion(row):
            """综合判断最终情感标签"""
            
            # 基于主导情感
            dominant = row.get('dominant_emotion', 'neutral')
            
            # 基于情感强度和极性的修正
            polarity = row.get('sentiment_polarity', 0)
            intensity = row.get('emotion_intensity', 0)
            
            # 情感映射规则
            if dominant == 'positive' or polarity > 0.1:
                if intensity > 0.15:
                    return 'happy'
                else:
                    return 'calm'  # 轻度正面情感归为平静
            elif dominant == 'negative' or polarity < -0.1:
                return 'sad'
            elif dominant == 'energetic':
                return 'energetic'
            elif dominant == 'calm':
                return 'calm'
            else:
                # 基于数值判断
                if polarity > 0.05:
                    return 'happy'
                elif polarity < -0.05:
                    return 'sad'
                elif intensity > 0.1:
                    return 'energetic'
                else:
                    return 'calm'
        
        # 生成最终情感标签
        self.df['emotion_label'] = self.df.apply(determine_final_emotion, axis=1)
        
        # 显示最终分布
        final_dist = self.df['emotion_label'].value_counts()
        print(f"最终情感标签分布:")
        for emotion, count in final_dist.items():
            print(f"   {emotion}: {count} 首 ({count/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def quality_control(self, min_text_length=20):
        """数据质量控制"""
        print(f"\n7. 数据质量控制")
        print("-" * 30)
        
        initial_count = len(self.df)
        
        # 过滤条件
        valid_mask = (
            (self.df['text_length'] >= min_text_length) &  # 最小文本长度
            (self.df['char_count'] >= 10) &  # 最小字符数
            (~self.df['processed_clean_text'].isin(['', None])) &  # 非空文本
            (self.df['title'].notna()) &  # 有标题
            (self.df['author'].notna())  # 有作者
        )
        
        self.df = self.df[valid_mask].copy()
        
        filtered_count = len(self.df)
        removed_count = initial_count - filtered_count
        
        print(f"   质量控制完成")
        print(f"   初始歌曲数: {initial_count}")
        print(f"   过滤后歌曲数: {filtered_count}")
        print(f"   移除歌曲数: {removed_count} ({removed_count/initial_count*100:.1f}%)")
        
        return self.df
    
    def add_additional_features(self):
        """添加额外特征"""
        print("\n8. 额外特征工程")
        print("-" * 30)
        
        # 文本复杂度特征
        self.df['text_complexity'] = (
            self.df.get('avg_hsk_level', 3.0) * 0.4 +
            (self.df['unique_chars'] / np.maximum(self.df['char_count'], 1)) * 0.3 +
            (self.df['char_density']) * 0.3
        )
        
        # 情感强度分级
        self.df['emotion_strength'] = pd.cut(
            self.df['emotion_intensity'], 
            bins=[0, 0.05, 0.15, 0.3, 1.0], 
            labels=['低', '中', '高', '极高']
        )
        
        # 歌曲长度分级
        self.df['song_length_category'] = pd.cut(
            self.df['text_length'], 
            bins=[0, 100, 300, 600, float('inf')], 
            labels=['短', '中', '长', '超长']
        )
        
        # 创建综合质量评分
        self.df['quality_score'] = (
            np.minimum(self.df['text_length'] / 500, 1.0) * 0.3 +  # 长度分
            np.minimum(self.df['total_meaningful_words'] / 100, 1.0) * 0.3 +  # 词汇量分
            (self.df['emotion_intensity'] * 5) * 0.2 +  # 情感丰富度分
            np.minimum(self.df.get('total_hsk_words', 50) / 100, 1.0) * 0.2  # HSK覆盖度分
        )
        
        print(f"   额外特征添加完成")
        print(f"   平均文本复杂度: {self.df['text_complexity'].mean():.2f}")
        print(f"   平均质量评分: {self.df['quality_score'].mean():.2f}")
        
        return self.df
    
    def visualize_data_insights(self):
        """数据洞察可视化"""
        print("\n9. 数据洞察可视化")
        print("-" * 30)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 情感分布饼图
        emotion_counts = self.df['emotion_label'].value_counts()
        axes[0, 0].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('情感标签分布')
        
        # 文本长度分布
        self.df['text_length'].hist(bins=50, ax=axes[0, 1])
        axes[0, 1].set_title('歌词长度分布')
        axes[0, 1].set_xlabel('字符数')
        
        # HSK难度分布
        if 'avg_hsk_level' in self.df.columns:
            self.df['avg_hsk_level'].hist(bins=30, ax=axes[0, 2])
            axes[0, 2].set_title('HSK难度分布')
            axes[0, 2].set_xlabel('平均HSK等级')
        
        # 情感极性分布
        axes[1, 0].hist(self.df['sentiment_polarity'], bins=50, alpha=0.7)
        axes[1, 0].set_title('情感极性分布')
        axes[1, 0].set_xlabel('情感极性分数')
        
        # 情感与文本长度关系
        emotion_length = self.df.groupby('emotion_label')['text_length'].mean()
        axes[1, 1].bar(emotion_length.index, emotion_length.values)
        axes[1, 1].set_title('各情感类型平均文本长度')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 质量评分分布
        self.df['quality_score'].hist(bins=30, ax=axes[1, 2])
        axes[1, 2].set_title('歌曲质量评分分布')
        axes[1, 2].set_xlabel('质量评分')
        
        plt.tight_layout()
        plt.show()
        
        print("数据可视化完成")
    
    def save_processed_data(self, output_file='processed_chinese_lyrics.csv', save_all_data=True):
        """保存处理后的数据"""
        print(f"\n10. 数据保存")
        print("-" * 30)
        
        if save_all_data:
            # 保存全部数据
            final_df = self.df.copy()
            save_count = len(final_df)
        else:
            # 只保存高质量数据（可选）
            high_quality_mask = (
                (self.df['quality_score'] >= 0.3) &
                (self.df['emotion_intensity'] >= 0.05)
            )
            final_df = self.df[high_quality_mask].copy()
            save_count = len(final_df)
        
        # 选择要保存的列
        columns_to_save = [
            # 基础信息
            'title', 'author', 'url', 'score',
            # 处理后的文本
            'processed_clean_text', 
            # 文本特征
            'text_length', 'char_count', 'line_count', 'unique_chars', 'char_density',
            # 情感特征
            'positive_count', 'negative_count', 'energetic_count', 'calm_count',
            'positive_ratio', 'negative_ratio', 'energetic_ratio', 'calm_ratio',
            'sentiment_polarity', 'emotion_intensity', 'dominant_emotion',
            'total_meaningful_words',
            # HSK特征（如果存在）
            'avg_hsk_level', 'hsk_complexity', 'total_hsk_words',
            # 最终标签和评分
            'emotion_label', 'text_complexity', 'quality_score'
        ]
        
        # 只保留存在的列
        existing_columns = [col for col in columns_to_save if col in final_df.columns]
        final_df = final_df[existing_columns]
        
        # 保存文件
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"   数据保存完成")
        print(f"   输出文件: {output_file}")
        print(f"   保存歌曲数: {save_count}")
        print(f"   特征维度: {len(existing_columns)}")
        print(f"   文件大小: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # 显示最终统计
        print(f"\n 最终数据集统计:")
        print(f"   情感分布: {final_df['emotion_label'].value_counts().to_dict()}")
        print(f"   平均文本长度: {final_df['text_length'].mean():.0f} 字符")
        print(f"   平均质量评分: {final_df['quality_score'].mean():.2f}")
        
        return final_df

def main():
    """主处理流程"""
    print("开始中文歌词数据集处理")
    
    # 初始化处理器
    processor = ChineseLyricsProcessor("chinese_lyrics.csv") 
    
    try:
        # 执行完整处理流程
        processor.load_data()
        processor.clean_lyrics_text()
        processor.extract_text_features()
        processor.analyze_emotion_keywords()
        processor.process_hsk_grades()
        processor.create_final_emotion_labels()
        processor.quality_control()
        processor.add_additional_features()
        
        # 可视化数据洞察
        processor.visualize_data_insights()
        
        # 保存处理后的数据（保存全部数据）
        final_df = processor.save_processed_data('processed_chinese_lyrics_full.csv', save_all_data=True)
        
        print("\n处理完成！")
    
        
        return processor, final_df
        
    except FileNotFoundError:
        print("找不到输入文件 'chinese_lyrics.csv'")
        print("请确保文件名和路径正确")
        return None, None
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    #主程序
    processor, processed_df = main()