import pickle
import json
import os

DATA_DIR = 'data'  # Change si tes fichiers sont ailleurs

TRAIN_FILE = os.path.join(DATA_DIR, 'train_data.pkl')
TEST_FILE = os.path.join(DATA_DIR, 'test_data.pkl')
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab.json')

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def decode_sentence(encoded, inv_vocab):
    return ' '.join([inv_vocab.get(str(idx), '<UNK>') for idx in encoded])

def inspect_data():
    print("🔍 Chargement des fichiers...")

    vocab = load_vocab(VOCAB_FILE)
    inv_vocab = {str(idx): tok for tok, idx in vocab.items()}

    train_data = load_pickle(TRAIN_FILE)
    test_data = load_pickle(TEST_FILE)

    print(f"\n📚 Taille du vocabulaire : {len(vocab)}")
    print(f"📊 Nombre de phrases dans train : {len(train_data)}")
    print(f"📊 Nombre de phrases dans test : {len(test_data)}")

    print("\n🔎 Exemples dans TRAIN DATA :\n")
    for i in range(3):
        print(f"{i+1}. Encodé : {train_data[i]}")
        print(f"   Décodé : {decode_sentence(train_data[i], inv_vocab)}\n")

    print("🔎 Exemples dans TEST DATA :\n")
    for i in range(3):
        print(f"{i+1}. Encodé : {test_data[i]}")
        print(f"   Décodé : {decode_sentence(test_data[i], inv_vocab)}\n")

if __name__ == '__main__':
    inspect_data()
