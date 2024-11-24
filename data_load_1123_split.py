import requests
import json
import os
import time

def write_json(file_path, docs, start):
    end = start + len(docs) -1
    file_path = f"{file_path.split('.json')[0]}_{start}_{end}.json"
    with open(file_path, 'w') as f:
        json.dump(docs, f, indent=4) 
    print(f"Split of unique documents saved: {len(docs)}")
    return end


def get_documents_revised(query, maxnum=10, max_disp_len=200, retries=5, backoff_factor=2):
    url = "https://api.infini-gram.io/"
    payload = {
        "index": "v4_dolma-v1_7_llama",
        "query_type": "search_docs",
        "query": query,
        "maxnum": maxnum,
        "max_disp_len": max_disp_len
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['documents']
        except requests.exceptions.HTTPError as e:
            wait_time = backoff_factor ** attempt  
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    print("Max retries reached. Failed to retrieve documents.")
    return []



def collect_unique_documents(query, file_path, json_num, batch_size=10, max_disp_len=200, target_count=10):
    
    seen_doc_ix = set([])
    all_documents = []
    print_cnt=0
    start=0

    while len(seen_doc_ix) < target_count:
        documents = get_documents_revised(query, maxnum=batch_size, max_disp_len=max_disp_len)
        if len(documents)==0: # getting HTTPS error
            break

        NEW = False
        for doc in documents:
            if doc['doc_ix'] not in seen_doc_ix:
                seen_doc_ix.add(doc['doc_ix']) # deduplicating same documents
                all_documents.append(doc)
                NEW = True


        if NEW:
            patience = 0
        else:
            patience += 1
            print(f"No new unique documents found.[{patience}]")
            if patience >= 15:
                break
            continue
        
        if len(all_documents)>=json_num:
            end = write_json(file_path, all_documents, start)
            all_documents = []
            start = end + 1

        # print logs
        print_cnt +=1
        if len(seen_doc_ix)%100==0 or print_cnt%40==0:
            print(f"Collected {len(seen_doc_ix)} unique documents so far...")
        time.sleep(1)

    _ = write_json(file_path, all_documents, start)    
    print(f"Document collection completed. Total unique documents saved: {len(seen_doc_ix)}")




def main():
    # expressions = {'is a made-up word':4700, 'is a made up word':4754, 'is not a common word':2479, 'is a created word':169, 'is an invented word':1977}
    # expressions = {'is a made up word':4754, 'is not a common word':2479, 'is a created word':169, 'is an invented word':1977}
    expressions = {'is a created word':169}

    DEBUG = False

    for expression in expressions:

        # json_num : number of part of documents to be saved
        json_num = 50 if DEBUG else expressions[expression]

        # target_count : number of total documents to load
        target_count = 150 if DEBUG else expressions[expression]


        collect_unique_documents(expression, f"./data/{'_'.join(expression.split())}.json", json_num=json_num, batch_size=10, max_disp_len=200, target_count=target_count)


def __init__():
    main()