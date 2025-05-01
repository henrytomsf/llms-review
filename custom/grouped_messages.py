from typing import List, Dict
import json
import sqlite3

db_path = '/Users/henrytom/Library/Messages/chat.db'

def fetch_messages():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    select 
        date,
        text,
        is_from_me
    from message 
    where handle_id = 45
        and not cache_has_attachments
        and is_delivered
        and type = 0
    order by date;
    """

    cursor.execute(query)
    messages = cursor.fetchall()
    conn.close()
    return messages

def group_adjacent_messages(messages):
    grouped = []
    current_sender = 0
    current_group = []

    for date, text, is_from_me in messages:
        if is_from_me != current_sender:
            if current_group != [None]:
                grouped.append((str(current_sender), ' '.join(current_group)))
            current_group = [text]
            current_sender = is_from_me
        else:
            current_group.append(text)

    # Don't forget the last group
    if current_group:
        grouped.append((str(current_sender), ' '.join(current_group)))

    return grouped

def create_pairings(grouped):
    json_pairs = []
    for i, grouped_obj in enumerate(grouped):
        is_from_me, _ = grouped_obj
        print("DEBUG: ", i, is_from_me, grouped[i], len(grouped))
        if (is_from_me == '1') and (i < len(grouped) - 1):
            print("PAIR")
            pair = {}
            pair['prompt'] = grouped[i][1]
            pair['completion'] = grouped[i+1][1]
            json_pairs.append(pair)
    return json_pairs

def write_jsonl(json_pairings: List[Dict[str, str]]):
    with open('data.jsonl', 'w') as f:
        for entry_pair in json_pairings:
            f.write(json.dumps(entry_pair) + '\n')

def main():
    messages = fetch_messages()
    grouped = group_adjacent_messages(messages)

    json_pairings = create_pairings(grouped)
    write_jsonl(json_pairings)

    return json_pairings

if __name__ == "__main__":
    json_pairings = main()
    print(json_pairings[-5::])
