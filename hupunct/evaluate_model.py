import json
import os
import pickle
import pandas as pd

from transformers import pipeline, AutoTokenizer
from datasets import Dataset, load_metric
from seqeval.metrics import classification_report

import pprint
from tqdm import tqdm

from functions import token_label_generator, tokenize_and_align_labels


mappings = {
    'none': 0,
    ',': 1,
    '.': 2,
    '!': 3,
    '?': 4,
    '-': 5,
    ':': 6,
    # "``": 7,
    # "''": 7,
    '+': 7,
    ',+': 8,
    '.+': 9,
    '!+': 10,
    '?+': 11,
    '-+': 12,
    ':+': 13,
    # "``+": 15,
    # "''+": 15,

}
label_names = [
    'O',
    'B-COMMA',
    'B-DOT',
    'B-EXCLAM',
    'B-QUES',
    'B-HYPHEN',
    'B-COLON',
    # 'B-QUOTE',
    'B-UPPER',
    'B-UPCOMMA',
    'B-UPDOT',
    'B-UPEXCLAM',
    'B-UPQUES',
    'B-UPHYPHEN',
    'B-UPCOLON',
    # 'B-UPQUOTE',
]

# mappings = {
#     'none': 0,
#     ',': 1,
#     '.': 2,
#     '!': 3,
#     '?': 4,
#     '-': 5,
#     ':': 6,
#     "``": 7,
#     "''": 7,
#     '+': 8,
#     ',+': 9,
#     '.+': 10,
#     '!+': 11,
#     '?+': 12,
#     '-+': 13,
#     ':+': 14,
#     "``+": 15,
#     "''+": 15,
#
# }
# label_names = [
#     'O',
#     'B-COMMA',
#     'B-DOT',
#     'B-EXCLAM',
#     'B-QUES',
#     'B-HYPHEN',
#     'B-COLON',
#     'B-QUOTE',
#     'B-UPPER',
#     'B-UPCOMMA',
#     'B-UPDOT',
#     'B-UPEXCLAM',
#     'B-UPQUES',
#     'B-UPHYPHEN',
#     'B-UPCOLON',
#     'B-UPQUOTE',
# ]

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


MODEL_CP = 'D:/00_DATA/02_ELTE/MODELS/hupunct-v02f-01/checkpoint-371200_final'
OUTPUT_DIR = 'D:/00_DATA/02_ELTE/TEST_SCORES'

MODEL_NAME = MODEL_CP.split('/')[-1]

tokenizer = AutoTokenizer.from_pretrained(MODEL_CP, model_max_len=512)

token_classifier = pipeline(
    "token-classification",
    tokenizer=tokenizer,
    model=MODEL_CP,
    aggregation_strategy="none",
    ignore_labels=[],
)

test_text_file_path = '../data/dataset_v2/test_data.txt'
# test_text_file_path = '../data/dataset_v2_2pc/test_data.txt'

fp_test = open(test_text_file_path, 'r', encoding='utf-8')

metric = load_metric('seqeval')


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:

            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:

            new_labels.append(-100)
        else:

            label = labels[word_id]
            new_labels.append(label)

    return new_labels


def map_words_to_string(examples):
    articles_str = []

    for tokens in examples['_tokens']:
        article_str = ' '.join(tokens)
        articles_str.append(article_str)

    examples['_tokens_str'] = articles_str

    return examples


def map_id2label(examples):
    all_labels = examples["labels"]

    labels_str = []

    for label in all_labels:
        label_str = [id2label[i] for i in label if i in id2label.keys()]

        labels_str.append(label_str)

    examples['_labels_str'] = labels_str

    return examples


def aggregate_preds(preds: dict):
    aggregated = []

    for i, pred in enumerate(preds[:-1]):
        word = pred.get('word')
        if '##' in word:
            continue

        next_word = preds[i+1].get('word')

        if '##' in next_word:
            done = False
            j = 1
            while not done and i+j+1<len(preds):
                word = word + next_word.replace('##', '')

                j += 1

                next_word = preds[i+j].get('word')
                if '##' not in next_word:
                    done = True

        aggregated.append({'entity': pred.get('entity'),
                           'prob': pred.get('score'),
                           'word': word})

    if '##' not in next_word:
        aggregated.append({'entity':  preds[i+1].get('entity'),
                           'prob':  preds[i+1].get('score'),
                           'word': next_word})

    return aggregated


ds_test = Dataset.from_generator(token_label_generator, gen_kwargs={'file_pointer': fp_test, 'mappings': mappings})

ds_test = ds_test.map(map_words_to_string, batched=True)
# ds_test_article = ds_test_article.rename_column('_labels', 'labels')
# ds_test_article = ds_test_article.rename_column('_tokens', 'tokens')

ds_test = ds_test.map(
    tokenize_and_align_labels,
    batched=True,
    load_from_cache_file=True,
    fn_kwargs={'tokenizer': tokenizer}
)
ds_test = ds_test.map(map_id2label, batched=True)

worst_examples = {999: {'tokens': [], 'labels': [], 'preds': {}}}
score_per_lens = {'lens': [], 'scores': []}

all_preds = []
all_trues = []

# step = 0
for example in tqdm(ds_test, desc='generating preds'):
    input_text = example['_tokens_str']
    true_labels = example['_labels_str']

    preds_dict = token_classifier(input_text)

    preds = [p.get('entity') for p in preds_dict]

    all_preds.append(preds)
    all_trues.append(true_labels)

    _scores = metric.compute(predictions=[preds], references=[true_labels])

    score_per_lens.get('lens').append(len(true_labels))
    score_per_lens.get('scores').append(_scores.get('overall_f1'))

    least_worst = max(worst_examples.keys())

    if len(list(worst_examples.keys())) < 1000:
        worst_examples.update({_scores.get('overall_f1'): {'text': input_text, 'labels': true_labels, 'pred': preds_dict}})

    elif _scores.get('overall_f1') < least_worst:
        _ = worst_examples.pop(least_worst)

        worst_examples.update({_scores.get('overall_f1'): {'text': input_text, 'labels': true_labels}})

    # print(scores)

    # if step>100:
    #     break
    # step+=1

# print(f'\nscoring for {len(all_preds)} examples\n')
scores = metric.compute(predictions=all_preds, references=all_trues)

pprint.pprint(scores)

for k in scores.keys():
    if type(scores.get(k)) == dict:
        scores.get(k).update({'number': int(scores.get(k).get('number'))})

out_name = 'test_scores'

os.makedirs(f'{OUTPUT_DIR}/{MODEL_NAME}', exist_ok=True)

with open(f'{OUTPUT_DIR}/{MODEL_NAME}/{out_name}.json', 'w') as f:
    json.dump(scores, f, indent=3)

cr = classification_report(all_trues, all_preds, output_dict=True)

cr_table = pd.DataFrame.from_dict(cr)

cr_table.to_parquet(f'{OUTPUT_DIR}/{MODEL_NAME}/{out_name}.parquet')
cr_table.to_csv(f'{OUTPUT_DIR}/{MODEL_NAME}/{out_name}.csv', header=True)
cr_table.to_excel(f'{OUTPUT_DIR}/{MODEL_NAME}/{out_name}.xlsx')

with open(f'{OUTPUT_DIR}/{MODEL_NAME}/{out_name}_worst_examples.pkl', 'wb') as f:
    pickle.dump(worst_examples, f)

with open(f'{OUTPUT_DIR}/{MODEL_NAME}/{out_name}_scores_per_lens.pkl', 'wb') as f:
    pickle.dump(score_per_lens, f)

print('\nsaved scores!:)\n')