import requests
import json
import os
import time

def get_documents(query, maxnum=10, max_disp_len=200, retries=10, backoff_factor=2, timeout=30):
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
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()['documents']
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 504]:  # 처리 제한 초과 및 게이트웨이 타임아웃 처리
                wait_time = backoff_factor ** attempt  
                print(f"Error {response.status_code}: Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                print(f"HTTP Error: {e}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
    
    print("Max retries reached. Failed to retrieve documents.")
    return None  

def collect_unique_documents(query, file_path, batch_size=10, max_disp_len=200, target_count=173269, max_duplicate_attempts=10):
    seen_doc_ix = set()
    all_documents = []
    duplicate_count = 0

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            all_documents = json.load(f)
        seen_doc_ix.update(doc['doc_ix'] for doc in all_documents)
        print(f"Loaded {len(all_documents)} existing documents from file.")

    while len(all_documents) < target_count:
        documents = get_documents(query, maxnum=batch_size, max_disp_len=max_disp_len)
        
        if documents is None:
            print("Failed to retrieve documents. Exiting.")
            break

        new_documents = [doc for doc in documents if doc['doc_ix'] not in seen_doc_ix]
        
        if not new_documents:
            duplicate_count += 1
            print(f"No new unique documents found in this batch. Duplicate attempt {duplicate_count} of {max_duplicate_attempts}.")
            
            if duplicate_count >= max_duplicate_attempts:
                print("Reached maximum duplicate attempts. Stopping collection.")
                break
        else:
            all_documents.extend(new_documents)
            seen_doc_ix.update(doc['doc_ix'] for doc in new_documents)
            duplicate_count = 0
            print(f"Collected {len(all_documents)} unique documents so far...")

        with open(file_path, 'w') as f:
            json.dump(all_documents, f, indent=4)
        time.sleep(1)
    
    print(f"Document collection completed. Total unique documents saved: {len(all_documents)}")

file_path = "/home/work/jupyter/minwoo/CMU/LLM_blocking/data/document/is_not_a_word_doc.json"

collect_unique_documents("is not a word", file_path, batch_size=10, max_disp_len=200, target_count=173269)
