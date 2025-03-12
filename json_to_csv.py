import os
import json
import pandas as pd
from tqdm import tqdm
import spacy
import re

# nltk.download('words')
# nltk.download('wordnet')   # Remove comments as needed


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
if 'sentencizer' not in nlp.pipe_names:
    nlp.add_pipe('sentencizer')


# 1. Convert JSON to CSV (unique span list per row)
def unique_spans_per_row(json_data, query, output_path):
    ixs, spans, metadatas, texts = [], [], [], []

    for document in tqdm(json_data):
        ix = document.get("doc_ix", [])
        span = document.get("spans", [])
        text = " ".join([s[0] for s in span])
        metadata = document.get("metadata", [])

        ixs.append(ix)
        spans.append(span)
        metadatas.append(metadata)
        texts.append(text)

    df = pd.DataFrame()
    df['doc_ix'] = ixs
    df['span'] = spans
    df['metadata'] = metadatas
    df['texts'] = texts
    df['query'] = [query] * len(df)

    dedup_df = df.drop_duplicates(subset=['doc_ix', 'metadata', 'texts']).reset_index(drop=True)
    dedup_df.index.name = 'span_id'
    dedup_df.reset_index(inplace=True)

    dedup_df.to_csv(output_path, index=False)
    print(f'Deduplicate dataframe {len(df)} into {len(dedup_df)}')
    print(f'Saved the csv file to {output_path}')

    return dedup_df


# 2. Extract **words** from each of the unique span list (CSV)
def extract_words_with_indices_from_span(span_list):
    
    # 1) Generate text by joining the elements of span_list with spaces
    cumulative_lengths = []
    total_length = 0

    for span in span_list:
        cumulative_lengths.append(total_length)
        total_length += len(span[0]) + 1  # +1: Space added during joining

    results = []
    # 2) Iterate through each span to find occurrences of 'expression(e.g., is not a real word)'
    for i in range(1, len(span_list)):
        if span_list[i][1] == '0':  # Check the 'is not a real word' condition
            # Extract Word : the last valid word from the previous span
            prev_word = None
            prev_text = span_list[i - 1][0].split()
            for p_i in range(len(prev_text)-1, -1, -1):
                if any(char.isalnum() for char in prev_text[p_i]):

                    prev_word = ' '.join(prev_text[p_i:]).lstrip()
                    break
            if prev_word is None:
                continue
            local_index = span_list[i - 1][0].rfind(prev_word)
            absolute_index = cumulative_lengths[i - 1] + local_index
            results.append((prev_word, absolute_index))


    return results


# 3. Use the spaCy pipeline to accurately extract sentences
def extract_pattern_sentence(word, pattern_index, doc, query):
    sents = list(doc.sents)  # Convert to a list for repeated use
    for i, sent in enumerate(sents):
        if sent.start_char <= pattern_index < sent.end_char:
            # Check if the current sentence contains the query and the word
            if query in sent.text.lower() and word in sent.text.lower():
                return sent.text.strip()
            else:
                # If there is a next sentence, concatenate and return both
                if i + 1 < len(sents):
                    combined = sent.text.strip() + " " + sents[i + 1].text.strip()
                    return combined
                else:
                    # If it is the last sentence, return the current sentence only
                    return sent.text.strip()
    return None




def main():

    os.makedirs("./data/csv", exist_ok=True)

    # Combine JSON files grouped by expressions into a single CSV file.

    expressions = ['is a made up word', 'is a made-up word', 'is a created word', 
                    'is not a common word', 'is an invented word', 
                    'is not a real word', 'is not a term', 'is not a word']

    # expressions = [expressions[-1]] # should be modified
    total_path = f"./data/csv/total.csv"    
    total = []

    for expression in expressions:
        json_path = f"./data/document/{'_'.join(expression.split())}.json"
        csv_path = f"./data/document/{'_'.join(expression.split())}.csv"
        csv_csv_path = f"./data/csv/{'_'.join(expression.split())}.csv"

        with open(json_path, "r") as file:
            data = json.load(file)
            print(f"Data count for '{expression}' : \t{len(data)}")


        df = unique_spans_per_row(data, expression, csv_path)
        

        # 4. Construct the final DataFrame 
        result_rows = []
        texts = df['texts'].tolist()
        docs = list(nlp.pipe(texts, batch_size=100)) # Batch processing using nlp.pipe()

        for row, doc in tqdm(zip(df.iterrows(), docs)):

            row = row[1]
            word_indices = extract_words_with_indices_from_span(row['span'])


            query = row.query

            for word, word_index in word_indices:
                word_ = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word.lower())

                sentence = extract_pattern_sentence(word_, word_index, doc, query)
                if sentence:
                    result_rows.append({
                        'span_id': row['span_id'],
                        'doc_ix': row.doc_ix,
                        'word': word,
                        'sentence': sentence,
                        'query': query
                    })


        result_df = pd.DataFrame(result_rows)
        result_df.dropna(inplace=True)
        result_df.reset_index(inplace=True, names='index')
        result_df.to_csv(csv_csv_path, index=False)
        print(f'original json{len(df)} into {len(result_df)}')
        total.append(result_df)
    
    # Merge all DataFrames into one
    total_df = pd.concat(total, ignore_index=True)
    total_df.to_csv(total_path, index=False)
    print(len(total_df))

        

if __name__ == "__main__":
    main()

