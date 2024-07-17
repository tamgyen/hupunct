import glob
import os
import numpy as np
import nltk
import random
import requests
import wandb

from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, get_cosine_schedule_with_warmup, AdamW
from datasets import Dataset, load_metric, DatasetDict

from tqdm.autonotebook import tqdm


# tokenizer = BertTokenizer.from_pretrained(
#     "../02_dependencies/hubert_wiki_lower")

metric = load_metric("seqeval")

eol_symbols = list(".,?!();:-%\"/'")

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

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


# **********************************************************************************************************************

def preprocess_line(s: str) -> str:
    for c in s:
        if not c in list(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,?!();:-–—%\"'„”’… séáőúűíöüóÉÁŐÚŰÍÖÜÓ\n"):
            # print(c, s)
            return ""
    s = s.replace('„', '"').replace('”', '"').replace('’', "'").replace('``', '"').replace('…', "...").replace('–',
                                                                                                               "-").replace(
        '—', "-").replace('\n', ' ')
    return s


def get_articles_from_raw(text_file_path: str, min_article_len: int = 500):
    articles = []
    article = ""

    i = 0
    with open(text_file_path, "r", encoding='utf-8') as fp:
        for line in fp:

            if line == '\n':
                if len(article) >= min_article_len:
                    articles.append(article[:-2])
                article = ""

            cleaned = preprocess_line(line)

            # check if there is punctuation in the line
            if cleaned != "":
                if (not (cleaned[0].isupper() or cleaned[0] in eol_symbols or cleaned[0].isnumeric())) or (
                        cleaned[-2] not in eol_symbols):
                    if len(article) >= min_article_len:
                        # check the beginning
                        while article[0].isnumeric() or article[0] == ' ' or article[0] in eol_symbols + ['-', '%',
                                                                                                          '/']:
                            article = article[1:]

                        articles.append(article)
                        article = ""
                        continue
                    else:
                        article = ""
                        continue

                else:
                    article += cleaned
                    i += 1

            # TODO
            # if i >= 20000:
            #     break

    return articles


def make_dataset_from_raw(folder_with_txts, output, splits):
    os.makedirs(output, exist_ok=True)

    files_raw = glob.glob(f'{folder_with_txts}/**/*.txt', recursive=True)

    random.shuffle(files_raw)

    fp_train = open(f'{output}/train_data.txt', "w", encoding='utf-8')
    fp_eval = open(f'{output}/eval_data.txt', "w", encoding='utf-8')
    fp_test = open(f'{output}/test_data.txt', "w", encoding='utf-8')

    fps = [fp_train, fp_eval, fp_test]

    num_all_articles = 0

    for file_raw in tqdm(files_raw, desc='generating datset'):
        articles = get_articles_from_raw(file_raw)

        for article in articles:
            random_choice = random.choices([0, 1, 2], weights=splits, k=1)[0]
            fps[random_choice].write(f"{article}\n")

            num_all_articles += 1

    return num_all_articles


# info: deprecated
def make_feature_target_token_class(articles: list[str]):
    features = []
    targets = []
    for article in tqdm(articles):

        words = nltk.word_tokenize(article)

        for j, word in enumerate(words[:-1]):
            if j == 0 and word in mappings.keys():
                continue

            if word not in mappings.keys() and words[j + 1] not in mappings.keys():

                if '-' in word:
                    split_word = word.split('-')
                    features.append(split_word[0].lower())
                    if split_word[1] != '':
                        features.append(split_word[1].lower())

                    if word[0].isupper():
                        targets.append(mappings.get('-+'))
                    else:
                        targets.append(mappings.get('-'))

                    if split_word[1] != '':
                        if split_word[1][0].isupper():
                            targets.append(mappings.get('+'))
                        else:
                            targets.append(mappings.get('none'))
                else:
                    # no -
                    features.append(word.lower())
                    if word[0].isupper():
                        targets.append(mappings.get('+'))
                    else:
                        targets.append(mappings.get('none'))

            elif word not in mappings.keys():
                if '-' in word:
                    split_word = word.split('-')
                    features.append(split_word[0].lower())

                    if split_word[1] != '':
                        features.append(split_word[1].lower())

                    if word[0].isupper():
                        targets.append(mappings.get('-+'))
                    else:
                        targets.append(mappings.get('-'))

                    if split_word[1] != '':
                        if split_word[1][0].isupper():
                            targets.append(mappings.get(words[j + 1] + '+'))
                        else:
                            targets.append(mappings.get(words[j + 1]))

                else:
                    # no -
                    features.append(word.lower())
                    if word[0].isupper():
                        targets.append(mappings.get(words[j + 1] + '+'))
                    else:
                        targets.append(mappings.get(words[j + 1]))

    if all(element in mappings.values() for element in targets):
        return features, targets
    else:
        print(targets)
        print(features)


# info: deprecated
def make_examples(features, targets, sequence_length: int = 128, increment: int = 64):
    assert len(features) == len(targets), f'features and targets len mismatch! check target generator'

    examples = {"tokens": [],
                "labels": []}

    start = 0
    while start < len(features) - sequence_length:
        examples.get("tokens").append(features[start:start + sequence_length])
        examples.get("labels").append(targets[start:start + sequence_length])

        start += increment

    return examples


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


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["_tokens"], is_split_into_words=True, truncation=True, max_length=512, padding="max_length"
    )
    all_labels = examples["_labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def token_label_generator(file_pointer, mappings):
    for article in file_pointer:

        features = []
        targets = []

        words = nltk.word_tokenize(article)

        for j, word in enumerate(words[:-1]):
            if j == 0 and word in mappings.keys():
                continue

            if word not in mappings.keys() and words[j + 1] not in mappings.keys():

                if '-' in word:
                    split_word = word.split('-')
                    features.append(split_word[0].lower())
                    if split_word[1] != '':
                        features.append(split_word[1].lower())

                    if word[0].isupper():
                        targets.append(mappings.get('-+'))
                    else:
                        targets.append(mappings.get('-'))

                    if split_word[1] != '':
                        if split_word[1][0].isupper():
                            targets.append(mappings.get('+'))
                        else:
                            targets.append(mappings.get('none'))
                else:
                    # no -
                    features.append(word.lower())
                    if word[0].isupper():
                        targets.append(mappings.get('+'))
                    else:
                        targets.append(mappings.get('none'))

            elif word not in mappings.keys():
                if '-' in word:
                    split_word = word.split('-')
                    features.append(split_word[0].lower())

                    if split_word[1] != '':
                        features.append(split_word[1].lower())

                    if word[0].isupper():
                        targets.append(mappings.get('-+'))
                    else:
                        targets.append(mappings.get('-'))

                    if split_word[1] != '':
                        if split_word[1][0].isupper():
                            targets.append(mappings.get(words[j + 1] + '+'))
                        else:
                            targets.append(mappings.get(words[j + 1]))

                else:
                    # no -
                    features.append(word.lower())
                    if word[0].isupper():
                        targets.append(mappings.get(words[j + 1] + '+'))
                    else:
                        targets.append(mappings.get(words[j + 1]))

        if not any(element is None for element in targets):
            yield {"_tokens": features, "_labels": targets}


def sequence_example(examples, increment=50, sequence_length=250):
    # sequence_length = random.randint(sequence_length_min, sequence_length_max)

    token_chunks = []
    label_chunks = []

    tokens_batch = examples['_tokens']
    labels_batch = examples['_labels']

    for tokens, labels in zip(tokens_batch, labels_batch):
        if sequence_length > len(tokens):
            token_chunks.append(tokens)
            label_chunks.append(labels)

        start = 0
        while start < len(tokens) - sequence_length:
            # examples.get("tokens").append(features[start:start + sequence_length])
            # examples.get("labels").append(targets[start:start + sequence_length])

            token_chunks.append(tokens[start:start + sequence_length])
            label_chunks.append(labels[start:start + sequence_length])

            # sequence_length = random.randint(sequence_length_min, sequence_length_max)

            start += increment

    examples['tokens'] = token_chunks
    examples['labels'] = label_chunks

    return examples


if __name__ == '__main__':
    pass
    # make_dataset_from_raw('D:/00_DATA/02_ELTE/dataset_full_raw', 'D:/00_DATA/02_ELTE/dataset_v3_5pc', splits=[.9, .1, .1])
    # train_text_file_path = '../data/dataset_mini/train_data.txt'
    # #
    # fp_train = open(train_text_file_path, 'r', encoding='utf-8')
    # # fp_eval = open(train_text_file_path.replace('train', 'eval'), 'r', encoding='utf-8')
    # #
    # ds_train = Dataset.from_generator(token_label_generator,
    #                                   gen_kwargs={'file_pointer': fp_train, 'mappings': mappings})
    #
    # for example in ds_train:
    #     print(example)
    #
    # ds_train_sequenced = ds_train.map(sequence_example,
    #                                   batched=True,
    #                                   remove_columns=ds_train.column_names,
    #                                   load_from_cache_file=True,
    #                                   fn_kwargs={'increment': increment, 'sequence_length': sequence_length}
    #                                   )
    #
    # ds_train_sequence_tokenized = ds_train_sequenced.map(
    #     tokenize_and_align_labels,
    #     batched=True,
    #     remove_columns=ds_train_sequenced.column_names,
    #     load_from_cache_file=True
    # )
    #
    # ds_eval = Dataset.from_generator(token_label_generator, gen_kwargs={'file_pointer': fp_eval, 'mappings': mappings})
    #
    #
    # ds_eval_sequenced = ds_eval.map(sequence_example,
    #                                 batched=True,
    #                                 remove_columns=ds_eval.column_names,
    #                                 load_from_cache_file=True,
    #                                 fn_kwargs={'increment': increment, 'sequence_length': sequence_length}
    #                                 )
    #
    # ds_eval_sequence_tokenized = ds_eval_sequenced.map(
    #     tokenize_and_align_labels,
    #     batched=True,
    #     remove_columns=ds_eval_sequenced.column_names,
    #     load_from_cache_file=True
    # )
    #
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
    #                                                    padding='max_length',
    #                                                    max_length=num_max_tokens,
    #                                                    )
    #
    # model = AutoModelForTokenClassification.from_pretrained(
    #     MODEL_CP,
    #     id2label=id2label,
    #     label2id=label2id,
    # )
    #
    # print(model.config.num_labels)
    #
    # args = TrainingArguments(
    #     MODEL_NAME,
    #     report_to=["wandb"],  # enable logging to W&B
    #     run_name=MODEL_NAME,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=2e-5,
    #     num_train_epochs=NUM_EP,
    #     weight_decay=0.01,
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=ds_train_sequence_tokenized,
    #     eval_dataset=ds_eval_sequence_tokenized,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    # )
    #
    # trainer.train()
    #
    # trainer.save_model(f"../04_models/{MODEL_NAME}")
