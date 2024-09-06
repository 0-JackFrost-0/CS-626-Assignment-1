import nltk
nltk.download("brown")
nltk.download("universal_tagset")

import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from copy import deepcopy

SEED = 42
np.random.seed(SEED)
sns.set_theme()

data = list(nltk.corpus.brown.tagged_sents(tagset="universal"))

SENTENCE_START = "^"
SENTENCE_START_POS = "-"

class HMM:
    def __init__(self, data, sentence_start=SENTENCE_START, sentence_start_pos=SENTENCE_START_POS, smoothing=1e-3):
        self.sentence_start = sentence_start
        self.sentence_start_pos = sentence_start_pos
        self.data = deepcopy(data)
        self.smoothing = smoothing
        self.process_data()

    def process_data(self):
        all_words = set()
        all_words.add(self.sentence_start)
        all_tags = set()
        all_tags.add(self.sentence_start_pos)

        for i, sentence in enumerate(self.data):
            for word, tag in sentence:
                all_words.add(word.lower())
                all_tags.add(tag)
            self.data[i] = [(self.sentence_start, self.sentence_start_pos)] + list(map(lambda x: (x[0].lower(), x[1]), sentence))
        
        self.all_words = list(all_words)
        self.all_words.sort()
        self.all_tags = list(all_tags)
        self.all_tags.sort()

        self.word_to_index = {word: i for i, word in enumerate(self.all_words)}
        self.index_to_word = {i: word for i, word in enumerate(self.all_words)}
        self.tag_to_index = {tag: i for i, tag in enumerate(self.all_tags)}
        self.index_to_tag = {i: tag for i, tag in enumerate(self.all_tags)}

    def train(self, train_data=None):
        self.transition_probs = np.zeros((len(self.all_tags), len(self.all_tags)))
        self.lexical_probs = np.zeros((len(self.all_tags), len(self.all_words)))

        if train_data is None:
            train_data = self.data

        for sentence in tqdm(train_data):
            self.lexical_probs[self.tag_to_index[self.sentence_start_pos]][self.word_to_index[self.sentence_start]] += 1

            for i in range(1, len(sentence)):
                word, tag = sentence[i]
                prev_tag = sentence[i - 1][1]

                self.transition_probs[self.tag_to_index[prev_tag]][self.tag_to_index[tag]] += 1
                self.lexical_probs[self.tag_to_index[tag]][self.word_to_index[word]] += 1

        self.transition_probs += self.smoothing
        self.lexical_probs += self.smoothing

        self.transition_probs /= np.sum(self.transition_probs, axis=1, keepdims=True)
        self.lexical_probs /= np.sum(self.lexical_probs, axis=1, keepdims=True)

    def viterbi(self, s):
        sentence_start_added = False

        if type(s) == str:
            s = s.lower().split()

        if s[0] != self.sentence_start:
            s = [self.sentence_start] + s
            sentence_start_added = True

        n_sent = len(s)
        n_tags = len(self.all_tags)

        best_probs = np.full((n_sent, n_tags), -np.inf) 
        best_tags = np.zeros((n_sent, n_tags), dtype=int)

        first_word = s[0]
        first_word_idx = self.word_to_index.get(first_word, None)

        for tag_idx in range(n_tags):
            best_probs[0][tag_idx] = np.log(self.lexical_probs[tag_idx][first_word_idx])
            best_tags[0][tag_idx] = 0

        for t in range(1, n_sent):
            word = s[t]
            word_idx = self.word_to_index.get(word, None)

            if word_idx is None:
                word_idx = -1

            for tag_idx in range(n_tags):
                max_prob = -np.inf
                best_prev_tag = None

                for prev_tag_idx in range(n_tags):
                    transition_prob = np.log(self.transition_probs[prev_tag_idx][tag_idx])
                    prob = best_probs[t-1][prev_tag_idx] + transition_prob

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag_idx

                if word_idx != -1:
                    lexical_prob = np.log(self.lexical_probs[tag_idx][word_idx])
                else:
                    lexical_prob = np.log(self.lexical_probs.min())  

                best_probs[t][tag_idx] = max_prob + lexical_prob
                best_tags[t][tag_idx] = best_prev_tag

        best_path = []
        best_last_tag = np.argmax(best_probs[-1])
        best_path.append(best_last_tag)

        for t in range(n_sent - 1, 0, -1):
            best_last_tag = best_tags[t][best_last_tag]
            best_path.insert(0, best_last_tag)

        best_tag_sequence = [self.index_to_tag[tag] for tag in best_path]

        if sentence_start_added:
            best_tag_sequence = best_tag_sequence[1:]

        return best_tag_sequence


    def evaluate(self, data):
        y_true = []
        y_pred = []

        all_tags = self.all_tags[1:]

        for sentence in tqdm(data):
            words = [word for word, _ in sentence]
            tags = [tag for _, tag in sentence]

            pred_tags = self.viterbi(words)

            y_true.extend(tags[1:])
            y_pred.extend(pred_tags[1:])

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=None,labels= all_tags)
        precision = precision_score(y_true, y_pred, average=None,labels= all_tags)
        recall = recall_score(y_true, y_pred, average=None,labels= all_tags)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision per tag:")
        for i, tag in enumerate(all_tags):
            print(f"{tag}: {precision[i]:.4f}" , end=" ")
        print("\nAverage Precision: ", np.mean(precision))
        print(f"Recall per tag:")
        for i, tag in enumerate(all_tags):
            print(f"{tag}: {recall[i]:.4f}" , end=" ")
        print("\nAverage Recall: ", np.mean(recall))
        print(f"F1 Score per tag:")
        for i, tag in enumerate(all_tags):
            print(f"{tag}: {f1[i]:.4f}", end=" ")
        print(f"\nAverage F1 Score: {np.mean(f1)}")

        print("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=all_tags, normalize="true")
        plt.figure(figsize=(15, 10))
        sns.heatmap(cm, annot=True,xticklabels=all_tags, yticklabels=all_tags, cmap = plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        return acc, f1, precision, recall

    def k_fold_cross_validation(self, k=5):
        acc_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        np.random.shuffle(self.data)
        n = len(self.data)
        fold_size = n // k

        for i in range(k):
            print(f"Fold {i + 1}")
            test_data = self.data[i * fold_size:(i + 1) * fold_size]
            train_data = self.data[:i * fold_size] + self.data[(i + 1) * fold_size:]

            self.train(train_data)
            acc, f1, precision, recall = self.evaluate(test_data)
            acc_list.append(acc)
            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)

        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)

        f1_score = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
        f_2_score = (5 * mean_precision * mean_recall) / (4 * mean_precision + mean_recall)
        f_half_score = (1.25 * mean_precision * mean_recall) / (0.25 * mean_precision + mean_recall)

        print(f"Overall average Accuracy: {np.mean(acc_list):.4f}")
        print(f"Overall average Precision: {np.mean(precision_list):.4f}")
        print(f"Overall average Recall: {np.mean(recall_list):.4f}")
        print(f"Overall average F1 Score: {f1_score}")
        print(f"Overall average F2 Score: {f_2_score:.4f}")
        print(f"Overall average F0.5 Score: {f_half_score:.4f}")


hmm = HMM(data) 
hmm.train()

def get_POS(s):
    return hmm.viterbi(s)


if __name__ == "__main__":
    hmm.k_fold_cross_validation()

    test_sentence = "The quick brown fox jumps over the lazy dog"
    print(f"POS tags for the sentence: {test_sentence}")
    print(get_POS(test_sentence))
