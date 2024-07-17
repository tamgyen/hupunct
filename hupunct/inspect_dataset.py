import json
import pickle

from functions import token_label_generator

mappings = {
    'none': 0,
    ',': 1,
    '.': 2,
    '!': 3,
    '?': 4,
    '-': 5,
    ':': 6,
    "``": 7,
    "''": 7,
    '+': 8,
    ',+': 9,
    '.+': 10,
    '!+': 11,
    '?+': 12,
    '-+': 13,
    ':+': 14,
    "``+": 15,
    "''+": 15,

}

fp_train = open('../data/dataset_v2/train_data.txt', 'r', encoding='utf-8')
fp_eval = open('../data/dataset_v2/eval_data.txt', 'r', encoding='utf-8')
fp_test = open('../data/dataset_v2/test_data.txt', 'r', encoding='utf-8')

fps = [fp_train]
names = ['train']

counts_all = 0
article_lengths = []

for fp, name in zip(fps, names):
    print(f'{name}\n')

    counts = {i: 0 for i in range(16)}
    j = 0
    for example in token_label_generator(fp, mappings):
        labels = example.get('_labels')

        [counts.update({l: counts.get(l) + 1}) for l in labels]

        article_lengths.append(len(labels))
        j += 1
        counts_all += 1

        if j % 1000 == 0:
            print(j)

    with open(f'../data/dataset_v2/stats/{name}_counts.json', 'w') as f:
        json.dump(counts, f, indent=3)

with open(f'../data/dataset_v2/stats/art_lens.pkl', 'wb') as f:
    pickle.dump(article_lengths, f)

print(f'all: {counts_all}')
