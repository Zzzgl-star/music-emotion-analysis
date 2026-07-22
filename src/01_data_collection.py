import pandas as pd
import numpy as np
import requests
import os

print("获取音乐数据集")
print("="*40)

def download_spotify_dataset():
    """下载Spotify音乐数据集"""
    
    print("下载Spotify音乐数据集...")
    
    # 包含32,000首歌曲
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
    
    try:
        # 直接下载CSV
        df = pd.read_csv(url)
        print(f"成功下载: {len(df)} 首真实歌曲")
        
        return df
        
    except Exception as e:
        print(f"下载失败: {e}")
        return None

def process_spotify_data(df):
    """处理Spotify数据集"""
    
    print("处理数据...")
    
    # 检查必需的列
    required_cols = ['track_name', 'track_artist', 'valence', 'energy', 'danceability']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"缺少必需列: {missing_cols}")
        return None
    
    # 清理数据
    df_clean = df.dropna(subset=required_cols).copy()
    
    # 基于真实音频特征添加情感标签
    def classify_emotion(row):
        valence = row['valence']
        energy = row['energy']
        danceability = row['danceability']
        
        # 基于音乐心理学的分类规则
        if valence > 0.7 and energy > 0.6:
            return 'happy'
        elif valence < 0.4 and energy < 0.5:
            return 'sad'
        elif energy > 0.7 and danceability > 0.6:
            return 'energetic'
        elif energy < 0.4 and valence > 0.3:
            return 'calm'
        else:
            # 基于主要特征决定
            if valence > 0.6:
                return 'happy'
            elif valence < 0.4:
                return 'sad'
            elif energy > 0.6:
                return 'energetic'
            else:
                return 'calm'
    
    df_clean['emotion_label'] = df_clean.apply(classify_emotion, axis=1)
    
    # 重命名列以保持一致性
    df_clean = df_clean.rename(columns={
        'track_name': 'title',
        'track_artist': 'artist',
        'track_album_name': 'album',
        'playlist_genre': 'genre'
    })
    
    # 选择最重要的列
    important_cols = [
        'title', 'artist', 'album', 'genre', 'emotion_label',
        'valence', 'energy', 'danceability', 'tempo', 'loudness',
        'acousticness', 'instrumentalness', 'speechiness', 'liveness'
    ]
    
    # 只保留存在的列
    available_cols = [col for col in important_cols if col in df_clean.columns]
    df_final = df_clean[available_cols].copy()
    
    print(f"数据处理完成: {len(df_final)} 首歌曲")
    print(f"情感分布: {df_final['emotion_label'].value_counts().to_dict()}")
    
    return df_final

def download_music_emotions_dataset():
    """下载音乐情感数据集"""
    
    print("\n下载音乐情感数据集...")
    
    urls = [
        "https://raw.githubusercontent.com/marshmellow77/music-emotion-prediction/main/data/emotions.csv",
        "https://raw.githubusercontent.com/enjuichang/PracticalDataScience-ENCA/main/data/Spotify_data.csv"
    ]
    
    for i, url in enumerate(urls):
        try:
            print(f"尝试数据源 {i+1}...")
            df = pd.read_csv(url)
            print(f"成功: {len(df)} 条记录")
            return df
        except:
            print(f"数据源 {i+1} 失败")
            continue
    
    print("所有情感数据集下载失败")
    return None

def get_real_music_dataset():
    """获取音乐数据集的主函数"""
    
    print("开始获取音乐数据集")
    print("="*50)
    
    #下载Spotify数据集
    spotify_df = download_spotify_dataset()
    
    if spotify_df is not None:
        # 处理数据
        processed_df = process_spotify_data(spotify_df)
        
        if processed_df is not None:
            # 平衡情感分布
            balanced_df = balance_emotions(processed_df)
            
            # 保存数据
            filename = 'music_emotion_dataset（1）.csv'
            balanced_df.to_csv(filename, index=False, encoding='utf-8')
            
            print(f"\n音乐数据集获取成功!")
            print(f"最终数据集:")
            print(f"文件: {filename}")
            print(f"歌曲数: {len(balanced_df)}")
            print(f"特征数: {len(balanced_df.columns)}")
            
            print(f"\n情感分布:")
            emotion_counts = balanced_df['emotion_label'].value_counts()
            for emotion, count in emotion_counts.items():
                print(f" {emotion}: {count} 首")
            
            print(f"\n数据预览:")
            print(balanced_df[['title', 'artist', 'emotion_label', 'valence', 'energy']].head())
            
            return balanced_df
        
    #尝试其他数据源
    print("\n尝试备用数据源...")
    emotions_df = download_music_emotions_dataset()

def balance_emotions(df, target_per_emotion=2000):
    """平衡情感分布"""
    
    print(f"\n平衡情感分布 (目标: 每种情感{target_per_emotion}首)")
    
    balanced_parts = []
    
    for emotion in df['emotion_label'].unique():
        emotion_df = df[df['emotion_label'] == emotion]
        
        if len(emotion_df) > target_per_emotion:
            # 如果超过目标数量，随机采样
            sampled = emotion_df.sample(n=target_per_emotion, random_state=42)
        else:
            # 如果不足，全部保留
            sampled = emotion_df
        
        balanced_parts.append(sampled)
        print(f"   {emotion}: {len(emotion_df)} -> {len(sampled)} 首")
    
    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    
    # 随机打乱顺序
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"平衡完成: 总计 {len(balanced_df)} 首歌曲")
    
    return balanced_df

def validate_real_dataset(df):
    """验证真实数据集"""
    
    print(f"\n数据集验证")
    print("-" * 30)
    
    print(f"数据形状: {df.shape}")
    print(f"数据类型: {df.dtypes.value_counts().to_dict()}")
    
    # 检查缺失值
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df) * 100).round(1)
    
    print(f"缺失值:")
    for col in df.columns:
        if missing_data[col] > 0:
            print(f"   {col}: {missing_data[col]} ({missing_percent[col]}%)")
    
    # 数值特征统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n数值特征范围:")
        for col in numeric_cols:
            if col in df.columns:
                print(f"   {col}: {df[col].min():.3f} - {df[col].max():.3f}")
    
    print("验证完成")

def main():
    """主函数"""
    
    print("开始获取真实音乐数据集...")
    
    # 获取真实数据
    dataset = get_real_music_dataset()
    
    if dataset is not None:
        # 验证数据
        validate_real_dataset(dataset)
        
        print(f"\n成功!")
        return dataset
    else:
        print("获取失败")
        return None

if __name__ == "__main__":
    # 运行主函数
    dataset = main()