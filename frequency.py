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
    return 0


def get_frequency(words):
    counts = []
    for word in tqdm(words):
        counts.append(get_count(word))
    return counts
    


def main():
    output_path = './data/freq/'
    df = pd.read_csv('./data/words/total_non_words.csv')
    
    # frequency within collected data
    collected = df['word'].value_counts()
    collected_df = collected.reset_index()
    collected_df.columns = ['word', 'count']
    collected_df.to_csv(f'{output_path}/collected.csv', index=False)

    # frequency within Dolma
    dolma_df = pd.DataFrame()
    dolma_df['word'] = collected_df['word'].copy()
    dolma_df['count'] = get_frequency(collected_df['word'])
    dolma_df = dolma_df.sort_values(by='count', ascending=False)
    dolma_df.to_csv(f'{output_path}/dolma.csv', index=False)




if __name__=='__main__':
    main()
