from sklearn import preprocessing
import numpy as np


class BagOfWords:
    def __init__(self):
        self.words = []
        self.vocab = []
        self.vocab_ln = 0

    def build_vocab (self, data):
        for doc in data:
            for word in doc:
                if word not in self.words:
                    self.words[word] = len(words)
                    self.words.append(word)
            self.vocab_ln = len(words)
            self.words = array.words

    def get_features(self, data):
        features = [[]]
        for dic in data:
            for nr_word in dic:
                contor = 0
                for doc in data:
                    ok = 0
                    for word in doc:
                        if ok!=0:
                            if word == nr_word:
                                contor += 1
                        ok +=1
                    features.append(contor)
        return features






def normalize_data ( train_sentences , test_sentences , type):
    if type == "standard":
        scaler = preprocessing.StandardScaler()
    elif type == "min_max":
        scaler = preprocessing.MinMaxScaler()
    elif type == "l2":
        scaler = preprocessing.Normalizer(norm ="l2")
    elif type == "l1":
        scaler = preprocessing.Normalizer(norm='l1')

    scaler.fit(train_sentences)
    print(scaler.mean_)
    print(scaler.scale_)
    scaled_x_train = scaler.transform(train_sentences)
    print(scaled_x_train)
    scaled_x_test = scaler.transform(test_sentences)
    print(scaled_x_test)
    return scaled_x_train



np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
train_sentences = np.load("training_sentences.npy")
train_labels = np.load("training_labels.npy")
train_labels = train_labels.astype(int)
test_sentences = np.load("test_sentences.npy")
test_labels = np.load("test_labels.npy")
test_labels = test_labels.astype(int)

Train = BagOfWords()
Test = BagOfWords()
train_features = BagOfWords()
test_features = BagOfWords()
Test.build_vocab(test_sentences)
Train.build_vocab(train_sentences)
Test.get_features(test_sentences)
Train.get_features(train_sentences)
normalize_data(train_sentences, test_sentences, "l2")