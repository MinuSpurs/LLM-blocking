import requests
import time
import pandas as pd
from tqdm import tqdm

def get_count(query, retries=5, backoff_factor=2):
    url = "https://api.infini-gram.io/"
    payload = {
        "index": "v4_dolma-v1_7_llama",
        "query_type": "count",
        "query": query,
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['count']
        except requests.exceptions.HTTPError as e:
            wait_time = backoff_factor ** attempt  
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    print("Max retries reached. Failed to retrieve documents.")
    return -1


def handle_failed_requests(word):

    time.sleep(1)
    cnt = get_count(word)
    return 0 if cnt == -1 else cnt


def get_frequency(words, orgs):
    counts = []
    
    for word, org in tqdm(zip(words, orgs)):
        cnts = 0  
        checked_words = set()  

        for w in [word, word.capitalize(), org]:

            # Avoid duplicate processing
            if w in checked_words:
                continue  
            checked_words.add(w)  
            
            cnt = get_count(w)
            if cnt == -1:  
                cnt = handle_failed_requests(w)
            
            cnts += cnt  
        
        counts.append(cnts)  
    
    return counts
    


def main():
    output_path = './data/freq/'
    df = pd.read_csv('./data/words/total_non_words.csv')

    # frequency within collected data
    collected = df['word'].value_counts()
    collected_df = collected.reset_index()
    collected_df.columns = ['word', 'count']

    collected_df['index'] = df['index']
    collected_df['span_id'] = df['span_id']
    collected_df['doc_ix'] = df['doc_ix']

    collected_df.to_csv(f'{output_path}/collected.csv', index=False)


    # frequency within Dolma
    dolma_df = pd.DataFrame()
    dolma_df['word'] = collected_df['word'].copy()

    # get original word to count
    org_word = df.groupby('word')['org'].first().to_dict()
    dolma_df['count'] = get_frequency(collected_df['word'], collected_df['word'].map(org_word))
    dolma_df = dolma_df.sort_values(by='count', ascending=False)
    
    dolma_df['index'] = df['index']
    dolma_df['span_id'] = df['span_id']
    dolma_df['doc_ix'] = df['doc_ix']
    
    dolma_df.to_csv(f'{output_path}/dolma.csv', index=False)

    print(len(df), len(collected_df), len(dolma_df))



if __name__=='__main__':
    main()
