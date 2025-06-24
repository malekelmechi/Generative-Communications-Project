# similarity.py
import numpy as np
from w3lib.html import remove_tags
from sklearn.preprocessing import normalize

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model  # ✅ Correction ici
from bert4keras.tokenizers import Tokenizer


class Similarity:
    def __init__(self, config_path, checkpoint_path, dict_path):
        self.model1 = build_transformer_model(config_path, checkpoint_path, with_pool=True)  # ✅ Correction ici
        self.model = keras.Model(
            inputs=self.model1.input,
            outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output
        )
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def compute_similarity(self, real_sentences, predicted_sentences):
        token_ids1, segment_ids1 = [], []
        token_ids2, segment_ids2 = [], []

        for sent1, sent2 in zip(real_sentences, predicted_sentences):
            sent1 = remove_tags(sent1)
            sent2 = remove_tags(sent2)

            ids1, sids1 = self.tokenizer.encode(sent1)
            ids2, sids2 = self.tokenizer.encode(sent2)

            token_ids1.append(ids1)
            token_ids2.append(ids2)
            segment_ids1.append(sids1)
            segment_ids2.append(sids2)

        token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
        token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')
        segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
        segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')

        vector1 = self.model.predict([token_ids1, segment_ids1])
        vector2 = self.model.predict([token_ids2, segment_ids2])

        vector1 = np.sum(vector1, axis=1)
        vector2 = np.sum(vector2, axis=1)

        vector1 = normalize(vector1, axis=0, norm='max')
        vector2 = normalize(vector2, axis=0, norm='max')

        dot = np.diag(np.matmul(vector1, vector2.T))  # a.b
        a = np.sqrt(np.diag(np.matmul(vector1, vector1.T)))  # ||a||
        b = np.sqrt(np.diag(np.matmul(vector2, vector2.T)))  # ||b||

        similarity_scores = dot / (a * b)
        return similarity_scores.tolist()
