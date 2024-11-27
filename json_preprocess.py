import json
import spacy
import pandas as pd
from tqdm import tqdm

from nltk.corpus import words, wordnet

import nltk
#nltk.download('words')
#nltk.download('wordnet')   # Remove comments as needed

nlp = spacy.load("en_core_web_sm")


# extracting a word and a sentence
def extract_data(data, output_file, query="is a made up word"):
    
    ixs, word_list, texts = [], [], []

    for document in data:
        spans = document.get("spans", [])
        ix = document.get("doc_ix", [])

        word = spans[0][0].split()[-1]
        text = " ".join([span[0] for span in spans])

        if word != [] and text != '':
            word_list.append(word.lower())
            texts.append(text)
            ixs.append(ix)

    docs = nlp.pipe(texts, batch_size=100)  
    
    sentences_with_query = [] 

    for doc in tqdm(docs):
        for sentence in doc.sents:
            if query in sentence.text:

                sentences_with_query.append(sentence.text.strip())
                break

    # Save csv file
    df = pd.DataFrame()
    df['word'] = word_list
    df['sentence'] = sentences_with_query
    df['doc_ix'] = ixs
    df.to_csv(output_file, index=False)
    print(f'Saved the csv file to {output_file}')

    return df


# csv filtering - only EN, length>3, not nltk word
def filter_words(df):
    english_words = set(words.words())

    filtered = df[df['word'].str.match('^[a-zA-Z]+$')] 
    filtered1 = filtered[filtered['word'].str.len()>3]
    filtered2 = filtered1[~filtered1['word'].isin(english_words)].copy()
    filtered3 = filtered2[filtered2['word'].apply(lambda x: len(wordnet.synsets(x))==0)].copy()

    return filtered3

def filter_archs(df, target_word):
    not_words = []

    for _, row in df.iterrows(): 
        sentence = row['sentence']
        doc_ix = row['doc_ix']
        doc = nlp(sentence)  # Process the sentence using spaCy

        for token in doc:
            # Check if the token matches the target word and is used as an attribute
            if token.text.lower() == target_word and token.dep_ == "attr":
                # Check for the "is not" structure
                if token.head.text.lower() == "is" and any(child.text.lower() == "not" for child in token.head.children):
                    # Extract "X" from the structure "X is not a word"
                    is_token_index = token.head.i
                    if is_token_index > 0:
                        x_token = doc[is_token_index - 1]  # Get the token representing "X"
                        x_word = x_token.text

                        # Append the word, sentence, and doc_ix to the result
                        print(f"Adding '{x_word}' to not_words list.")
                        not_words.append((x_word, sentence, doc_ix))  # Include doc_ix in the tuple

    # Convert the not_words list into a DataFrame
    filtered_df = pd.DataFrame(not_words, columns=['not_word', 'sentence', 'doc_ix'])
    return filtered_df


def main():

    # expressions = ['is a made-up word', 'is a made up word', 'is not a common word', 'is a created word', 'is an invented word']
    total = []

    expressions = ['is a made up word', 'is not a common word', 'is a created word', 'is an invented word'] # add expressions as needed
    for expression in expressions:

        target_word = expression.split()[-1]

        json_path = "/home/work/jupyter/minwoo/CMU/LLM_blocking/data/document/is_not_a_real_word_doc.json"
        csv_path = f"./data/csv/{'_'.join(expression.split())}.csv"
        
        with open(json_path, "r") as file:
            data = json.load(file)
            print(f"Data count for '{expression}' : \t{len(data)}")
        
        df = extract_data(data, csv_path, query=expression)

        df1 = filter_words(df)

        df1 = filter_archs(df1, target_word)
        df1.to_csv(f"{csv_path.split('.csv')[0]}_filtered.csv", index=False)
        print(f'Saving filtered csv file with len {len(df1)}')

        total.append(df1)

    csv_path = f"./data/words/total_non_words.csv"
    total_df = pd.concat(total, ignore_index=True)
    total_df.to_csv(csv_path, index=False)




main()

