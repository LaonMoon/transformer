import json

# train_english_file_path = 'datasets/iwslt17.de.en/train.de-en.en'
# train_german_file_path = 'datasets/iwslt17.de.en/train.de-en.de'
# test_english_file_path = 'datasets/iwslt17.de.en/valid.de-en.en'
# test_german_file_path = 'datasets/iwslt17.de.en/valid.de-en.de'

type_list = ['train', 'valid']

for dataset_type in type_list:
    with open(f'datasets/iwslt17.de.en/{dataset_type}.de-en.en', 'r', encoding='utf-8') as eng_file:
        english_lines = eng_file.readlines()

    with open(f'datasets/iwslt17.de.en/{dataset_type}.de-en.de', 'r', encoding='utf-8') as ger_file:
        german_lines = ger_file.readlines()

    data = []
    for eng, ger in zip(english_lines, german_lines):
        entry = {
            "input": ger.strip(),
            "output": eng.strip()
        }
        data.append(entry)

    with open(f'{dataset_type}_de_en.json', 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)