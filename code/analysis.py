import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Define the circle of fifths order
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

# Create a mapping from pitch classes to their positions in the circle of fifths
PITCH_CLASS_TO_POSITION = {pc: idx for idx, pc in enumerate(CIRCLE_OF_FIFTHS)}

# Number of pitch classes
N_PITCH_CLASSES = len(CIRCLE_OF_FIFTHS)

# Calculate the angle for each pitch class
ANGLES = {pc: (2 * np.pi * idx) / N_PITCH_CLASSES for pc, idx in PITCH_CLASS_TO_POSITION.items()}

def angle_to_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])

# Create vectors for each pitch class
PITCH_CLASS_VECTORS = {pc: angle_to_vector(angle) for pc, angle in ANGLES.items()}

# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def normalize_cosine_similarity(cosine_similarity):
    """
    Normalize cosine similarity from [-1, 1] to [0, 1].
    """
    return (cosine_similarity + 1) / 2
    
def normalize_audio_feats(feats_df, tempo_factor = 1, include_categorical=True, circle_5 = True):
    # Normalize continuous features
    scaler = MinMaxScaler()
    continuous_features = ['tempo', 'valence', 'liveness', 'instrumentalness', 'acousticness', 'speechiness', 'loudness', 'energy', 'danceability']
    # feats_df[continuous_features] = scaler.fit_transform(feats_df[continuous_features])
    scaler.fit(feats_df[continuous_features])
    feats_df.tempo = tempo_factor*feats_df.tempo
    feats_df[continuous_features] = scaler.transform(feats_df[continuous_features])

    # Encode categorical features
    if include_categorical:
        if circle_5: 
            categorical_encoded = np.array([PITCH_CLASS_VECTORS[k] for k in feats_df['key'].tolist()])
            categorical_encoded_df = pd.DataFrame(categorical_encoded)
            categorical_encoded_df.columns = ['C5_0', 'C5_1']
        else:
            encoder = OneHotEncoder(sparse_output=False)
            categorical_features = feats_df[['key']]
            categorical_encoded = encoder.fit_transform(categorical_features)
            categorical_encoded_df = pd.DataFrame(categorical_encoded)
        
        # # Combine all features
        feats_df = pd.concat([feats_df[continuous_features], categorical_encoded_df], axis=1)
    # print(feats_df)    
    # scaler = MinMaxScaler()
    # feats_df = pd.DataFrame(scaler.fit_transform(feats_df))
    # feats_df.columns = continuous_features+['C5_0', 'C5_1']
    # print(feats_df)
    return feats_df

def compare_seeds_rec(seed_audio_feats_df, rec_audio_feats_df, include_categorical=True, circle_5 = True):
    n_seeds = len(seed_audio_feats_df)
    audio_feats_df = pd.concat([seed_audio_feats_df, rec_audio_feats_df], axis=0, ignore_index=True)
    audio_feats_df = normalize_audio_feats(audio_feats_df, include_categorical, circle_5)
    d = cosine_similarity(audio_feats_df.iloc[n_seeds:,:], audio_feats_df.iloc[:n_seeds,:])
    d_max = np.max(d, axis=1)
    return d_max

def compare_discog(audio_feats_df_1, audio_feats_df_2, include_categorical=True, circle_5 = True):
    n_seeds = len(seed_audio_feats_df)
    audio_feats_df = pd.concat([seed_audio_feats_df, rec_audio_feats_df], axis=0, ignore_index=True)
    audio_feats_df = normalize_audio_feats(audio_feats_df, include_categorical, circle_5)
    d = cosine_similarity(audio_feats_df.iloc[n_seeds:,:], audio_feats_df.iloc[:n_seeds,:])
    d_max = np.max(d, axis=1)
    return d_max