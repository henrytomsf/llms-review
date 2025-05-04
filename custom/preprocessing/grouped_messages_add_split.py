import json


def add_split(filename: str):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    final_data = []
    for pair in data:
        pair['split'] = 'train'
        final_data.append(pair)

    with open('data_with_split.jsonl', 'w') as f:
        for entry_pair in final_data:
            f.write(json.dumps(entry_pair) + '\n')



if __name__ == '__main__':
    add_split(filename='../data/data.jsonl')
