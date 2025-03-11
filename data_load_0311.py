import requests
import json
import os
import time
import csv
import pandas as pd




def find_documents(query, retries=5, backoff_factor=2, timeout=30):

    url = "https://api.infini-gram.io/"
    payload = {
        "index": "v4_dolma-v1_7_llama",
        "query_type": "find",
        "query": query  
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[find_documents] Request failed (attempt {attempt+1}): {e}")
            if hasattr(e, "response") and e.response is not None:
                print("Response content:", e.response.text)
            time.sleep(backoff_factor ** attempt)

    print("Max retries reached. Failed to retrieve document locations.")
    return None 

def get_document_by_rank(s, rank, query, max_disp_len=200, retries=5, backoff_factor=2, timeout=30):
    url = "https://api.infini-gram.io/"
    payload = {
        "index": "v4_dolma-v1_7_llama",
        "query_type": "get_doc_by_rank",
        "s": s,                   
        "rank": rank,
        "max_disp_len": max_disp_len,
        "query": query            
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # print(f"[get_document_by_rank] Request failed (attempt {attempt+1}): {e}")
            # if hasattr(e, "response") and e.response is not None:
            #     print("Response content:", e.response.text)
            time.sleep(backoff_factor ** (attempt+1))

    return None  

def collect_documents_in_order(query, file_path, batch_size=10, max_disp_len=200):
    all_documents = []
    temp_batch = []
    rank_again = []
    rank_check = []

    os.makedirs(os.path.dirname(file_path), exist_ok=True)


    # Check the location of document from find()
    find_result = find_documents(query)
    if not find_result or 'segment_by_shard' not in find_result:
        print(f"⚠ No documents found using `find()` for '{query}'. Exiting.")
        return

    total_occurrences = sum(end - start for start, end in find_result['segment_by_shard'])
    print(f"Total occurrences in segments: {total_occurrences}")

    for s, (start, end) in enumerate(find_result['segment_by_shard']):
        
        print(f"🔍 Processing shard {s}: ranks {start} to {end}")
        for rank in range(start, end):
            doc = get_document_by_rank(s=s, rank=rank, query=query, max_disp_len=max_disp_len)
            if doc:       
                temp_batch.append(doc)
                rank_check.append((s, rank))

                if len(temp_batch) >= batch_size:
                    all_documents.extend(temp_batch)
                    temp_batch = []
                    with open(file_path, 'w') as f:
                        json.dump(all_documents, f, indent=4)
                    print(f"Documents saved so far : {len(all_documents)}")
            else:
                rank_again.append((s, rank))
                time.sleep(1)
        time.sleep(1)

    if temp_batch:
        all_documents.extend(temp_batch)
        with open(file_path, 'w') as f:
            json.dump(all_documents, f, indent=4)
    
    print(f"[Completed] Collected documents so far : {len(all_documents)} | Should be done again : {len(rank_again)}")
    return rank_check, rank_again

def check_json_file_count(file_path):
    # Print the count of saved documents
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"JSON file '{file_path}' contains {len(data)} documents.")
    else:
        print(f"JSON file '{file_path}' does not exist.")


def handle_failed_requests(expression, file_path, ranks):
    
    with open(file_path, 'r') as f:
        docs = json.load(f)

    rank_again_ = []
    temp_batch = []   
    for s, r in ranks:
        doc = get_document_by_rank(s=s, rank=r, query=query, max_disp_len=200)
        if doc:
            temp_batch.append(doc)
        else:
            rank_again_.append((s,r))
    docs.extend(temp_batch)

    with open(file_path, 'w') as f:
        json.dump(docs, f, indent=4)
    print(f"[Final] expression '{expression}'| Retrieved : {len(docs)}, Not retrieved : {len(rank_again_)}")
    
    if rank_again_:
        nr_path = os.path.join(os.path.dirname(file_path), 'not_retrieved')
        os.makedirs(nr_path, exist_ok=True)
        with open(f'{nr_path}/{expression}.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rank_again_)

## SET EXPRESSION LIST
# expressions = ['is not a real word', 'is a made-up word', 'is a made up word', 'is not a common word', 'is an invented word', 'is a created word', 'is not a term']
# expressions = ['is not a term']

ex_df = pd.read_csv('./data/expressions.csv')
expressions = list(ex_df[ex_df['1']!='o']['expression'])

# import pdb;pdb.set_trace()
for expression in expressions:
    print(expression)
    exp = '_'.join(expression.split(' '))
    file_path = f"./data/document/{exp}.json"
    query = expression
    # import pdb;pdb.set_trace()
    rank_check, rank_again = collect_documents_in_order(query, file_path, batch_size=50, max_disp_len=200)
    
    if rank_again:
        time.sleep(5)
        handle_failed_requests(expression, file_path, rank_again)

    check_json_file_count(file_path)
