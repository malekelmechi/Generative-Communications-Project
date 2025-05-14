import os
import re
import pickle
import json
from tqdm import tqdm
from collections import defaultdict

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

INPUT_DATA_DIRS = [
    'data/txt/en',
    'data/txt/fr'
]
OUTPUT_TRAIN_DIR = 'data/train_data.pkl'
OUTPUT_TEST_DIR = 'data/test_data.pkl'
OUTPUT_VOCAB_DIR = 'data/vocab.json'
MIN_LENGTH = 4
MAX_LENGTH = 30

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    # Conserver les lettres accentuées, œ, ç, etc. ainsi que certains signes de ponctuation utiles
    text = re.sub(r"[^a-zA-ZÀ-ÿœŒçÇ.'!?\-]+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

def filter_sentences(sentences, min_length=MIN_LENGTH, max_length=MAX_LENGTH):
    filtered_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if min_length <= len(words) <= max_length:
            filtered_sentences.append(' '.join(words))
    return filtered_sentences

def build_vocab(sentences, special_tokens=SPECIAL_TOKENS):
    token_to_count = defaultdict(int)
    for sentence in sentences:
        for word in sentence.split():
            token_to_count[word] += 1

    vocab = {token: idx for token, idx in special_tokens.items()}
    for word, count in sorted(token_to_count.items()):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Sauvegardé : {file_path}')

def preprocess_data(input_dirs, output_train_dir, output_test_dir, output_vocab_dir):
    sentences = []

    print('Nettoyage des fichiers texte...')
    for input_dir in input_dirs:
        for filename in tqdm(os.listdir(input_dir)):
            if not filename.endswith('.txt'):
                continue
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                cleaned_sentences = [clean_text(line) for line in text.split('\n') if line.strip()]
                sentences.extend(cleaned_sentences)

    print('Filtrage des phrases...')
    sentences = filter_sentences(sentences)

    print('Construction du vocabulaire...')
    vocab = build_vocab(sentences)
    print(f'Taille du vocabulaire : {len(vocab)}')

    print('Encodage des phrases...')
    encoded_sentences = []
    for sentence in tqdm(sentences):
        tokens = ['<START>'] + sentence.split() + ['<END>']
        encoded_sentence = [vocab.get(word, vocab['<UNK>']) for word in tokens]
        encoded_sentences.append(encoded_sentence)

    print('Division des données...')
    split_idx = int(len(encoded_sentences) * 0.8)
    train_data = encoded_sentences[:split_idx]
    test_data = encoded_sentences[split_idx:]

    save_data(train_data, output_train_dir)
    save_data(test_data, output_test_dir)
    with open(output_vocab_dir, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    print(f'Sauvegardé : {output_vocab_dir}')

if __name__ == '__main__':
    preprocess_data(INPUT_DATA_DIRS, OUTPUT_TRAIN_DIR, OUTPUT_TEST_DIR, OUTPUT_VOCAB_DIR)
