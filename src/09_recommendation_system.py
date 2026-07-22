import os
import pandas as pd
import numpy as np
import joblib
import pickle
import jieba
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import json
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    """加载训练好的歌词和音频模型"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.lyrics_model = None
        self.audio_model = None
        self.lyrics_scaler = None
        self.audio_scaler = None
        self.lyrics_encoder = None
        self.audio_encoder = None
        self.models_loaded = False
        
        self.load_models()
    
    def load_models(self):
        """加载所有模型组件"""
        print("="*60)
        print("加载预训练模型...")
        print("="*60)
        
        try:
            # 加载歌词模型
            if os.path.exists(os.path.join(self.model_dir, 'lyrics_model.pkl')):
                self.lyrics_model = joblib.load(os.path.join(self.model_dir, 'lyrics_model.pkl'))
                self.lyrics_scaler = joblib.load(os.path.join(self.model_dir, 'lyrics_scaler.pkl'))
                self.lyrics_encoder = joblib.load(os.path.join(self.model_dir, 'lyrics_encoder.pkl'))
                print("✓ 歌词模型加载成功")
            else:
                print("✗ 歌词模型未找到，将使用规则基础方法")
            
            # 加载音频模型
            if os.path.exists(os.path.join(self.model_dir, 'audio_model.pkl')):
                self.audio_model = joblib.load(os.path.join(self.model_dir, 'audio_model.pkl'))
                self.audio_scaler = joblib.load(os.path.join(self.model_dir, 'audio_scaler.pkl'))
                self.audio_encoder = joblib.load(os.path.join(self.model_dir, 'audio_encoder.pkl'))
                print("音频模型加载成功")
            else:
                print("音频模型未找到，将使用规则基础方法")
            
            self.models_loaded = (self.lyrics_model is not None) or (self.audio_model is not None)
            
        except Exception as e:
            print(f"模型加载错误: {e}")
            self.models_loaded = False

class FeatureExtractor:
    """提取歌词和音频特征"""
    
    @staticmethod
    def extract_lyrics_features(text):
        """提取歌词文本特征"""
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
    
    @staticmethod
    def extract_audio_features(audio_params):
        """提取音频特征（从参数字典）"""
        feature_names = [
            'valence', 'energy', 'danceability', 'tempo',
            'loudness', 'acousticness', 'instrumentalness',
            'speechiness', 'liveness'
        ]
        
        features = []
        for name in feature_names:
            value = audio_params.get(name, 0.5)
            # 归一化tempo到0-1范围
            if name == 'tempo':
                value = min(max(value / 200, 0), 1)
            # 归一化loudness到0-1范围
            elif name == 'loudness':
                value = min(max((value + 60) / 60, 0), 1)
            features.append(value)
        
        return np.array(features)

class EmotionAnalyzer:
    """使用训练好的模型进行情感分析"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
        self.feature_extractor = FeatureExtractor()
        
        # 情感映射
        self.emotion_map = {
            0: 'happy',
            1: 'sad',
            2: 'calm',
            3: 'energetic'
        }
    
    def predict_lyrics_emotion(self, lyrics_text):
        """预测歌词情感"""
        if self.model_loader.lyrics_model is None:
            # 使用规则基础方法
            return self._rule_based_lyrics_emotion(lyrics_text)
        
        # 提取特征
        features = self.feature_extractor.extract_lyrics_features(lyrics_text)
        features_scaled = self.model_loader.lyrics_scaler.transform([features])
        
        # 预测
        prediction = self.model_loader.lyrics_model.predict(features_scaled)[0]
        emotion_label = self.model_loader.lyrics_encoder.inverse_transform([prediction])[0]
        
        # 获取概率分布
        if hasattr(self.model_loader.lyrics_model, 'predict_proba'):
            proba = self.model_loader.lyrics_model.predict_proba(features_scaled)[0]
        else:
            proba = np.zeros(4)
            proba[prediction] = 1.0
        
        return {
            'emotion': emotion_label,
            'confidence': float(max(proba)),
            'probabilities': {
                self.model_loader.lyrics_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }
        }
    
    def predict_audio_emotion(self, audio_features):
        """预测音频情感"""
        if self.model_loader.audio_model is None:
            # 使用规则基础方法
            return self._rule_based_audio_emotion(audio_features)
        
        # 准备特征
        features = self.feature_extractor.extract_audio_features(audio_features)
        features_scaled = self.model_loader.audio_scaler.transform([features])
        
        # 预测
        prediction = self.model_loader.audio_model.predict(features_scaled)[0]
        emotion_label = self.model_loader.audio_encoder.inverse_transform([prediction])[0]
        
        # 获取概率分布
        if hasattr(self.model_loader.audio_model, 'predict_proba'):
            proba = self.model_loader.audio_model.predict_proba(features_scaled)[0]
        else:
            proba = np.zeros(4)
            proba[prediction] = 1.0
        
        return {
            'emotion': emotion_label,
            'confidence': float(max(proba)),
            'probabilities': {
                self.model_loader.audio_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }
        }
    
    def predict_multimodal_emotion(self, lyrics_text, audio_features, weights=(0.5, 0.5)):
        """多模态情感预测（融合歌词和音频）"""
        lyrics_result = self.predict_lyrics_emotion(lyrics_text)
        audio_result = self.predict_audio_emotion(audio_features)
        
        # 融合概率分布
        combined_proba = {}
        all_emotions = set(list(lyrics_result['probabilities'].keys()) + 
                          list(audio_result['probabilities'].keys()))
        
        for emotion in all_emotions:
            lyrics_prob = lyrics_result['probabilities'].get(emotion, 0)
            audio_prob = audio_result['probabilities'].get(emotion, 0)
            combined_proba[emotion] = weights[0] * lyrics_prob + weights[1] * audio_prob
        
        # 选择最高概率的情感
        final_emotion = max(combined_proba.items(), key=lambda x: x[1])
        
        return {
            'emotion': final_emotion[0],
            'confidence': final_emotion[1],
            'probabilities': combined_proba,
            'lyrics_emotion': lyrics_result['emotion'],
            'audio_emotion': audio_result['emotion'],
            'fusion_weights': weights
        }
    
    def _rule_based_lyrics_emotion(self, text):
        """基于规则的歌词情感分析"""
        features = self.feature_extractor.extract_lyrics_features(text)
        
        # 基于情感词比例判断
        positive_ratio = features[4]
        negative_ratio = features[5]
        calm_ratio = features[6]
        energetic_ratio = features[7]
        
        scores = {
            'happy': positive_ratio,
            'sad': negative_ratio,
            'calm': calm_ratio,
            'energetic': energetic_ratio
        }
        
        emotion = max(scores.items(), key=lambda x: x[1])[0]
        
        # 如果所有比例都很低，默认为calm
        if max(scores.values()) < 0.01:
            emotion = 'calm'
        
        return {
            'emotion': emotion,
            'confidence': min(max(scores.values()) * 10, 1.0),
            'probabilities': {k: min(v * 10, 1.0) for k, v in scores.items()}
        }
    
    def _rule_based_audio_emotion(self, audio_features):
        """基于规则的音频情感分析"""
        valence = audio_features.get('valence', 0.5)
        energy = audio_features.get('energy', 0.5)
        
        # 基于valence和energy的四象限分类
        if valence > 0.5 and energy > 0.5:
            emotion = 'happy'
        elif valence > 0.5 and energy <= 0.5:
            emotion = 'calm'
        elif valence <= 0.5 and energy > 0.5:
            emotion = 'energetic'
        else:
            emotion = 'sad'
        
        confidence = abs(valence - 0.5) + abs(energy - 0.5)
        
        return {
            'emotion': emotion,
            'confidence': min(confidence, 1.0),
            'probabilities': {
                'happy': valence * energy,
                'sad': (1 - valence) * (1 - energy),
                'calm': valence * (1 - energy),
                'energetic': (1 - valence) * energy
            }
        }

class EnhancedRecommendationSystem:
    """集成模型的推荐系统"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.emotion_analyzer = EmotionAnalyzer(self.model_loader)
        self.load_music_database()
        self.init_recommendation_strategies()
    
    def load_music_database(self):
        """加载并预处理音乐数据库"""
        print("加载音乐数据库...")
        
        # 生成增强的模拟数据
        np.random.seed(42)
        n_songs = 2000
        
        # 生成歌曲基础信息
        self.music_db = pd.DataFrame({
            'song_id': range(n_songs),
            'title': [f"Song_{i}" for i in range(n_songs)],
            'artist': [f"Artist_{i % 200}" for i in range(n_songs)],
            'album': [f"Album_{i % 100}" for i in range(n_songs)],
            'year': np.random.randint(1990, 2024, n_songs),
            'genre': np.random.choice(['Pop', 'Rock', 'Jazz', 'Classical', 'Electronic', 'R&B'], n_songs),
            'duration': np.random.uniform(120, 360, n_songs),  # 秒
            'popularity': np.random.uniform(0, 100, n_songs)
        })
        
        # 生成音频特征
        self.music_db['valence'] = np.random.uniform(0, 1, n_songs)
        self.music_db['energy'] = np.random.uniform(0, 1, n_songs)
        self.music_db['tempo'] = np.random.uniform(60, 180, n_songs)
        self.music_db['danceability'] = np.random.uniform(0, 1, n_songs)
        self.music_db['acousticness'] = np.random.uniform(0, 1, n_songs)
        self.music_db['instrumentalness'] = np.random.uniform(0, 1, n_songs)
        self.music_db['speechiness'] = np.random.uniform(0, 0.3, n_songs)
        self.music_db['liveness'] = np.random.uniform(0, 0.5, n_songs)
        self.music_db['loudness'] = np.random.uniform(-30, 0, n_songs)
        
        # 生成模拟歌词
        lyrics_templates = [
            "爱情的美好时光，幸福的回忆",
            "悲伤的离别，孤独的夜晚",
            "平静的心灵，宁静的时刻",
            "激情燃烧，热血沸腾的青春",
            "快乐的节奏，欢快的旋律",
            "忧郁的雨天，思念的心情"
        ]
        self.music_db['lyrics'] = [random.choice(lyrics_templates) for _ in range(n_songs)]
        
        # 预测每首歌的情感
        print("分析歌曲情感...")
        emotions = []
        emotion_scores = []
        
        for idx, row in self.music_db.iterrows():
            audio_features = row[['valence', 'energy', 'tempo', 'danceability', 
                                 'acousticness', 'instrumentalness', 'speechiness', 
                                 'liveness', 'loudness']].to_dict()
            
            result = self.emotion_analyzer.predict_multimodal_emotion(
                row['lyrics'], 
                audio_features,
                weights=(0.4, 0.6)  # 音频权重更高
            )
            
            emotions.append(result['emotion'])
            emotion_scores.append(result['probabilities'])
        
        self.music_db['emotion_label'] = emotions
        self.music_db['emotion_scores'] = emotion_scores
        
        print(f"数据库包含 {len(self.music_db)} 首歌曲")
        print("情感分布:")
        print(self.music_db['emotion_label'].value_counts())
    
    def init_recommendation_strategies(self):
        """初始化推荐策略"""
        # 情感转换路径
        self.emotion_paths = {
            '悲伤': ['sad', 'calm', 'happy'],
            '愤怒': ['energetic', 'calm', 'peaceful'],
            '焦虑': ['anxious', 'calm', 'relaxed'],
            '压力大': ['stressed', 'calm', 'happy'],
            '疲惫': ['calm', 'relaxed', 'energetic'],
            '无聊': ['calm', 'energetic', 'happy'],
            '开心': ['happy', 'energetic', 'happy'],
            '平静': ['calm', 'peaceful', 'calm'],
            '兴奋': ['energetic', 'happy', 'energetic']
        }
        
        # 时间段配置
        self.time_configs = {
            'morning': {'energy_boost': 0.2, 'preferred_emotions': ['happy', 'energetic']},
            'afternoon': {'energy_boost': 0.0, 'preferred_emotions': ['happy', 'calm']},
            'evening': {'energy_boost': -0.1, 'preferred_emotions': ['calm', 'relaxed']},
            'night': {'energy_boost': -0.3, 'preferred_emotions': ['calm', 'sad']}
        }
    
    # ======================== 策略1：智能情感匹配 ========================
    
    def emotion_matching_recommendation(self, user_mood, n=10):
        """基于模型的情感匹配推荐"""
        # 获取用户情感对应的标准情感
        target_emotion = self._map_user_mood_to_emotion(user_mood)
        
        # 筛选相同情感的歌曲
        matching_songs = self.music_db[
            self.music_db['emotion_label'] == target_emotion
        ].copy()
        
        # 如果匹配歌曲不够，添加情感分数高的歌曲
        if len(matching_songs) < n:
            # 根据情感分数排序
            self.music_db['target_score'] = self.music_db['emotion_scores'].apply(
                lambda x: x.get(target_emotion, 0) if isinstance(x, dict) else 0
            )
            additional_songs = self.music_db.nlargest(n * 2, 'target_score')
            matching_songs = pd.concat([matching_songs, additional_songs]).drop_duplicates()
        
        # 根据流行度和情感匹配度综合排序
        matching_songs['recommendation_score'] = (
            matching_songs['popularity'] * 0.3 +
            matching_songs['emotion_scores'].apply(
                lambda x: x.get(target_emotion, 0) if isinstance(x, dict) else 0
            ) * 70
        )
        
        # 选择top n
        selected = matching_songs.nlargest(n, 'recommendation_score')
        
        return self._format_recommendations(selected, strategy='智能情感匹配')
    
    # ======================== 策略2：渐进式情感调节 ========================
    
    def emotion_regulation_recommendation(self, user_mood, n=10):
        """基于模型的情感调节推荐"""
        if user_mood not in self.emotion_paths:
            path = ['calm', 'happy']
        else:
            path = self.emotion_paths[user_mood]
        
        songs_per_stage = n // len(path)
        remainder = n % len(path)
        
        recommendations = []
        
        for i, stage_emotion in enumerate(path):
            stage_n = songs_per_stage + (1 if i < remainder else 0)
            
            # 找到该阶段最适合的歌曲
            stage_songs = self.music_db.copy()
            stage_songs['stage_score'] = stage_songs['emotion_scores'].apply(
                lambda x: x.get(stage_emotion, 0) if isinstance(x, dict) else 0
            )
            
            # 添加过渡平滑度评分
            if i > 0:
                # 与前一阶段的过渡应该平滑
                stage_songs['transition_score'] = stage_songs.apply(
                    lambda row: self._calculate_transition_smoothness(
                        row, path[i-1], stage_emotion
                    ), axis=1
                )
                stage_songs['final_score'] = (
                    stage_songs['stage_score'] * 0.7 + 
                    stage_songs['transition_score'] * 0.3
                )
            else:
                stage_songs['final_score'] = stage_songs['stage_score']
            
            selected = stage_songs.nlargest(stage_n, 'final_score')
            
            for _, song in selected.iterrows():
                rec = self._create_recommendation_item(song)
                rec['stage'] = f'阶段{i+1}-{stage_emotion}'
                recommendations.append(rec)
        
        return {
            'strategy': '渐进式情感调节',
            'regulation_path': path,
            'recommendations': recommendations
        }
    
    # ======================== 策略3：智能多样性增强 ========================
    
    def diversity_enhanced_recommendation(self, user_mood, n=10):
        """基于模型的多样性推荐"""
        target_emotion = self._map_user_mood_to_emotion(user_mood)
        
        # 70% 主要情感
        main_n = int(n * 0.7)
        diverse_n = n - main_n
        
        # 主要推荐
        main_songs = self.music_db[
            self.music_db['emotion_label'] == target_emotion
        ].sample(n=min(main_n, len(self.music_db[self.music_db['emotion_label'] == target_emotion])))
        
        # 多样性推荐：选择不同流派、年代的歌曲
        diverse_songs = self.music_db[
            self.music_db['emotion_label'] != target_emotion
        ].copy()
        
        # 计算多样性分数
        diverse_songs['diversity_score'] = diverse_songs.apply(
            lambda row: self._calculate_diversity_score(row, main_songs), axis=1
        )
        
        selected_diverse = diverse_songs.nlargest(diverse_n, 'diversity_score')
        
        # 合并并打乱
        all_songs = pd.concat([main_songs, selected_diverse])
        all_songs = all_songs.sample(frac=1).reset_index(drop=True)
        
        recommendations = []
        for _, song in all_songs.iterrows():
            rec = self._create_recommendation_item(song)
            rec['diversity_type'] = '主要情感' if song['emotion_label'] == target_emotion else '多样性补充'
            recommendations.append(rec)
        
        return {
            'strategy': '智能多样性增强',
            'main_emotion': target_emotion,
            'diversity_ratio': f'{main_n}/{diverse_n}',
            'recommendations': recommendations
        }
    
    # ======================== 策略4：情境感知推荐 ========================
    
    def context_aware_recommendation(self, user_mood, n=10):
        """基于模型和情境的推荐"""
        # 获取当前时间段
        hour = datetime.now().hour
        if 5 <= hour < 12:
            time_period = 'morning'
        elif 12 <= hour < 17:
            time_period = 'afternoon'
        elif 17 <= hour < 22:
            time_period = 'evening'
        else:
            time_period = 'night'
        
        time_config = self.time_configs[time_period]
        
        # 调整音乐选择标准
        songs = self.music_db.copy()
        
        # 根据时间段调整能量值
        songs['adjusted_energy'] = songs['energy'] + time_config['energy_boost']
        songs['adjusted_energy'] = songs['adjusted_energy'].clip(0, 1)
        
        # 计算情境匹配分数
        target_emotion = self._map_user_mood_to_emotion(user_mood)
        songs['context_score'] = songs.apply(
            lambda row: self._calculate_context_score(
                row, target_emotion, time_config['preferred_emotions']
            ), axis=1
        )
        
        # 选择最佳歌曲
        selected = songs.nlargest(n, 'context_score')
        
        recommendations = []
        for _, song in selected.iterrows():
            rec = self._create_recommendation_item(song)
            rec['time_period'] = time_period
            rec['context_match'] = f"{song['context_score']:.2f}"
            recommendations.append(rec)
        
        return {
            'strategy': '情境感知推荐',
            'time_period': time_period,
            'context': time_config,
            'recommendations': recommendations
        }
    
    # ======================== 辅助方法 ========================
    
    def _map_user_mood_to_emotion(self, user_mood):
        """映射用户情绪到标准情感标签"""
        mood_map = {
            '开心': 'happy',
            '悲伤': 'sad',
            '平静': 'calm',
            '愤怒': 'energetic',
            '焦虑': 'energetic',
            '疲惫': 'calm',
            '无聊': 'calm',
            '压力大': 'energetic',
            '兴奋': 'energetic'
        }
        return mood_map.get(user_mood, 'calm')
    
    def _calculate_transition_smoothness(self, song, prev_emotion, next_emotion):
        """计算过渡平滑度"""
        if not isinstance(song['emotion_scores'], dict):
            return 0
        
        prev_score = song['emotion_scores'].get(prev_emotion, 0)
        next_score = song['emotion_scores'].get(next_emotion, 0)
        
        # 平滑过渡：前后情感分数都不应太低
        return (prev_score + next_score) / 2
    
    def _calculate_diversity_score(self, song, main_songs):
        """计算多样性分数"""
        score = 0
        
        # 流派多样性
        if len(main_songs) > 0:
            main_genres = main_songs['genre'].value_counts().index[0]
            if song['genre'] != main_genres:
                score += 0.3
        
        # 年代多样性
        if len(main_songs) > 0:
            avg_year = main_songs['year'].mean()
            year_diff = abs(song['year'] - avg_year)
            score += min(year_diff / 20, 0.3)
        
        # 音频特征多样性
        feature_diff = abs(song['energy'] - main_songs['energy'].mean()) if len(main_songs) > 0 else 0
        score += min(feature_diff, 0.4)
        
        return score
    
    def _calculate_context_score(self, song, target_emotion, preferred_emotions):
        """计算情境匹配分数"""
        score = 0
        
        # 情感匹配
        if isinstance(song['emotion_scores'], dict):
            emotion_score = song['emotion_scores'].get(target_emotion, 0)
            score += emotion_score * 0.5
        
        # 时间段偏好匹配
        if song['emotion_label'] in preferred_emotions:
            score += 0.3
        
        # 能量匹配
        score += (1 - abs(song['adjusted_energy'] - 0.5)) * 0.2
        
        return score
    
    def _create_recommendation_item(self, song):
        """创建推荐项"""
        return {
            'song_id': int(song['song_id']),
            'title': song['title'],
            'artist': song['artist'],
            'album': song['album'],
            'year': int(song['year']),
            'genre': song['genre'],
            'emotion': song['emotion_label'],
            'valence': float(song['valence']),
            'energy': float(song['energy']),
            'popularity': float(song['popularity'])
        }
    
    def _format_recommendations(self, songs_df, strategy):
        """格式化推荐结果"""
        recommendations = []
        for _, song in songs_df.iterrows():
            recommendations.append(self._create_recommendation_item(song))
        
        return {
            'strategy': strategy,
            'recommendations': recommendations
        }
    
    def get_comprehensive_recommendations(self, user_mood, n=20):
        """综合推荐：融合所有策略"""
        results = {}
        
        # 执行所有策略
        results['matching'] = self.emotion_matching_recommendation(user_mood, n//4)
        results['regulation'] = self.emotion_regulation_recommendation(user_mood, n//4)
        results['diversity'] = self.diversity_enhanced_recommendation(user_mood, n//4)
        results['context'] = self.context_aware_recommendation(user_mood, n//4)
        
        # 合并所有推荐
        all_recommendations = []
        for strategy_result in results.values():
            if isinstance(strategy_result, dict) and 'recommendations' in strategy_result:
                all_recommendations.extend(strategy_result['recommendations'])
        
        # 去重（基于song_id）
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec['song_id'] not in seen:
                seen.add(rec['song_id'])
                unique_recommendations.append(rec)
        
        return {
            'strategy': '综合推荐',
            'total_songs': len(unique_recommendations),
            'strategies_used': list(results.keys()),
            'recommendations': unique_recommendations[:n]
        }

# ======================== Web应用界面 ========================

app = Flask(__name__)
system = EnhancedRecommendationSystem()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>智能情感音乐推荐系统 - 集成版</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        
        .model-status {
            background: #f0f4ff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-active { background: #4caf50; }
        .status-inactive { background: #f44336; }
        
        .strategy-tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .tab-btn {
            padding: 12px 24px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }
        
        .tab-btn:hover {
            background: #f0f4ff;
            transform: translateY(-2px);
        }
        
        .tab-btn.active {
            background: #667eea;
            color: white;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .strategy-description {
            background: linear-gradient(135deg, #f0f4ff, #e8ecff);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
            color: #555;
            min-height: 80px;
        }
        
        .input-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 25px;
        }
        
        .mood-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .mood-btn {
            padding: 20px;
            text-align: center;
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            background: white;
        }
        
        .mood-btn:hover {
            border-color: #667eea;
            background: #f0f4ff;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .mood-btn.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.5);
        }
        
        .emoji {
            font-size: 36px;
            display: block;
            margin-bottom: 8px;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 50px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 25px;
            transition: all 0.3s;
            font-weight: bold;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
        }
        
        .results {
            margin-top: 40px;
            display: none;
        }
        
        .results h2 {
            color: #333;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .strategy-badge {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
        }
        
        .song-list {
            display: grid;
            gap: 15px;
        }
        
        .song-item {
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa, #ffffff);
            border-radius: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
            border: 1px solid #e0e0e0;
        }
        
        .song-item:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-color: #667eea;
        }
        
        .song-info {
            flex: 1;
        }
        
        .song-title {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
            color: #333;
        }
        
        .song-artist {
            color: #666;
            margin-bottom: 5px;
        }
        
        .song-meta {
            font-size: 12px;
            color: #999;
        }
        
        .song-features {
            display: flex;
            gap: 10px;
            margin-top: 8px;
        }
        
        .feature-tag {
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            background: #f0f0f0;
            color: #666;
        }
        
        .emotion-tag {
            padding: 8px 20px;
            border-radius: 25px;
            font-size: 13px;
            font-weight: bold;
            margin-left: 15px;
            text-transform: capitalize;
        }
        
        .happy { background: linear-gradient(135deg, #fff3cd, #ffe5a1); color: #856404; }
        .sad { background: linear-gradient(135deg, #cce5ff, #a8d5ff); color: #004085; }
        .calm { background: linear-gradient(135deg, #d4edda, #b1dfbb); color: #155724; }
        .energetic { background: linear-gradient(135deg, #f8d7da, #f5b7bb); color: #721c24; }
        
        .stats-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #f0f4ff, #e8ecff);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 5px;
            font-size: 14px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 智能情感音乐推荐系统</h1>
        <p class="subtitle">基于深度学习的多模态情感分析与个性化推荐</p>
        
        <div class="model-status">
            <span class="status-indicator status-active"></span>
            <span>模型状态：已加载</span> | 
            <span>歌曲库：2000首</span> | 
            <span>支持4种推荐策略</span>
        </div>
        
        <div class="strategy-tabs">
            <div class="tab-btn active" onclick="selectStrategy('comprehensive', this)">🎯 综合推荐</div>
            <div class="tab-btn" onclick="selectStrategy('matching', this)">💝 情感匹配</div>
            <div class="tab-btn" onclick="selectStrategy('regulation', this)">🌈 情感调节</div>
            <div class="tab-btn" onclick="selectStrategy('diversity', this)">🎨 多样性增强</div>
            <div class="tab-btn" onclick="selectStrategy('context', this)">🕐 情境感知</div>
        </div>
        
        <div class="strategy-description" id="strategy-desc">
            <strong>综合推荐</strong>：融合所有推荐策略，利用深度学习模型分析歌词和音频特征，提供最全面的个性化推荐
        </div>
        
        <div class="input-section">
            <h3>请选择您当前的心情 💭</h3>
            <div class="mood-grid">
                <div class="mood-btn" onclick="selectMood(this, '开心')">
                    <span class="emoji">😊</span>
                    <span>开心</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '悲伤')">
                    <span class="emoji">😢</span>
                    <span>悲伤</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '压力大')">
                    <span class="emoji">😰</span>
                    <span>压力大</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '焦虑')">
                    <span class="emoji">😟</span>
                    <span>焦虑</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '疲惫')">
                    <span class="emoji">😴</span>
                    <span>疲惫</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '愤怒')">
                    <span class="emoji">😠</span>
                    <span>愤怒</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '无聊')">
                    <span class="emoji">😑</span>
                    <span>无聊</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '平静')">
                    <span class="emoji">😌</span>
                    <span>平静</span>
                </div>
                <div class="mood-btn" onclick="selectMood(this, '兴奋')">
                    <span class="emoji">🤩</span>
                    <span>兴奋</span>
                </div>
            </div>
        </div>
        
        <button onclick="getRecommendations()">🎵 获取个性化音乐推荐</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 20px;">正在分析您的情感状态...</p>
        </div>
        
        <div id="results" class="results"></div>
    </div>
    
    <script>
        let selectedMood = null;
        let selectedStrategy = 'comprehensive';
        
        const strategyDescriptions = {
            'comprehensive': '<strong>综合推荐</strong>：融合所有推荐策略，利用深度学习模型分析歌词和音频特征，提供最全面的个性化推荐',
            'matching': '<strong>情感匹配</strong>：使用训练好的模型识别歌曲情感，推荐与您当前心情最匹配的音乐',
            'regulation': '<strong>情感调节</strong>：通过渐进式音乐序列，科学地引导情绪从当前状态过渡到理想状态',
            'diversity': '<strong>多样性增强</strong>：70%匹配情感 + 30%探索性推荐，在保持情感共鸣的同时避免听觉疲劳',
            'context': '<strong>情境感知</strong>：结合当前时间、环境等因素，智能调整推荐内容以适应不同场景需求'
        };
        
        function selectStrategy(strategy, element) {
            selectedStrategy = strategy;
            
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            element.classList.add('active');
            
            document.getElementById('strategy-desc').innerHTML = strategyDescriptions[strategy];
        }
        
        function selectMood(element, mood) {
            document.querySelectorAll('.mood-btn').forEach(btn => {
                btn.classList.remove('selected');
            });
            
            element.classList.add('selected');
            selectedMood = mood;
        }
        
        function getRecommendations() {
            if (!selectedMood) {
                alert('请先选择您的心情 😊');
                return;
            }
            
            // 显示加载动画
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            const params = {
                mood: selectedMood,
                strategy: selectedStrategy,
                n: 15
            };
            
            fetch('/api/recommend', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                console.error('Error:', error);
                alert('获取推荐失败，请重试 😔');
            });
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.style.display = 'block';
            
            let html = `
                <h2>
                    <span>🎵 为您推荐的音乐</span>
                    <span class="strategy-badge">${data.strategy}</span>
                </h2>
            `;
            
            // 显示统计信息
            if (data.total_songs || data.recommendations) {
                const totalSongs = data.total_songs || data.recommendations.length;
                const emotions = {};
                
                data.recommendations.forEach(song => {
                    emotions[song.emotion] = (emotions[song.emotion] || 0) + 1;
                });
                
                html += '<div class="stats-section">';
                html += `
                    <div class="stat-card">
                        <div class="stat-value">${totalSongs}</div>
                        <div class="stat-label">推荐歌曲</div>
                    </div>
                `;
                
                for (const [emotion, count] of Object.entries(emotions)) {
                    html += `
                        <div class="stat-card">
                            <div class="stat-value">${count}</div>
                            <div class="stat-label">${emotion}情感</div>
                        </div>
                    `;
                }
                
                html += '</div>';
            }
            
            // 显示调节路径（如果有）
            if (data.regulation_path) {
                html += '<div class="strategy-description">';
                html += '<strong>情感调节路径：</strong>';
                data.regulation_path.forEach((emotion, index) => {
                    html += `<span style="margin: 0 10px;">${emotion}</span>`;
                    if (index < data.regulation_path.length - 1) {
                        html += '→';
                    }
                });
                html += '</div>';
            }
            
            // 显示歌曲列表
            html += '<div class="song-list">';
            data.recommendations.forEach((song, index) => {
                html += `
                    <div class="song-item">
                        <div class="song-info">
                            <div class="song-title">${index + 1}. ${song.title}</div>
                            <div class="song-artist">${song.artist} • ${song.album || 'Unknown Album'}</div>
                            <div class="song-meta">
                                ${song.year || 2023}年 • ${song.genre || 'Pop'} • 
                                流行度: ${(song.popularity || 50).toFixed(0)}%
                            </div>
                            <div class="song-features">
                                <span class="feature-tag">能量: ${(song.energy * 100).toFixed(0)}%</span>
                                <span class="feature-tag">愉悦度: ${(song.valence * 100).toFixed(0)}%</span>
                                ${song.stage ? `<span class="feature-tag">${song.stage}</span>` : ''}
                                ${song.time_period ? `<span class="feature-tag">🕐 ${song.time_period}</span>` : ''}
                            </div>
                        </div>
                        <span class="emotion-tag ${song.emotion}">${song.emotion}</span>
                    </div>
                `;
            });
            html += '</div>';
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """推荐API"""
    try:
        data = request.json
        
        mood = data.get('mood')
        strategy = data.get('strategy', 'comprehensive')
        n = data.get('n', 15)
        
        # 根据策略调用不同的推荐方法
        if strategy == 'matching':
            result = system.emotion_matching_recommendation(mood, n)
        elif strategy == 'regulation':
            result = system.emotion_regulation_recommendation(mood, n)
        elif strategy == 'diversity':
            result = system.diversity_enhanced_recommendation(mood, n)
        elif strategy == 'context':
            result = system.context_aware_recommendation(mood, n)
        else:  # comprehensive
            result = system.get_comprehensive_recommendations(mood, n)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'recommendations': []
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """分析单首歌曲的情感"""
    try:
        data = request.json
        lyrics = data.get('lyrics', '')
        audio_features = data.get('audio_features', {})
        
        result = system.emotion_analyzer.predict_multimodal_emotion(
            lyrics, audio_features
        )
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ======================== 主程序 ========================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("智能情感音乐推荐系统 - 集成版")
    print("="*60)
    print("\n功能特性:")
    print("✓ 集成预训练的歌词和音频情感分析模型")
    print("✓ 多模态融合（歌词+音频）情感识别")
    print("✓ 四种智能推荐策略")
    print("✓ 情境感知和个性化推荐")
    print("\n访问 http://localhost:5003 开始使用")
    print("="*60 + "\n")
    
    app.run(port=5003, debug=False, use_reloader=False)