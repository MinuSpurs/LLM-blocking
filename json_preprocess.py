import json
import spacy
import pandas as pd
from tqdm import tqdm
from utils import *
import Levenshtein
from nltk.corpus import words, wordnet
import nltk
import os
import re 
import requests
import wikipedia


# nltk.download('words')
# nltk.download('wordnet')   # Remove comments as needed


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


# csv filtering - only EN, 23>length>3, 3> consecutive letters, not nltk word, wordnet
def filter_words(df):
    english_words = set(words.words())
    consecutive_pattern = r'(.)\1{2,}' 

    filtered = df[df['word'].str.match('^[a-zA-Z]+$')]
    filtered1 = filtered[filtered['word'].str.len().between(4, 22)]
    filtered2 = filtered1[filtered1['word'].apply(lambda x: not bool(re.search(consecutive_pattern, x)))]
    filtered2 = filtered2[~filtered2['word'].isin(english_words)].copy()
    filtered3 = filtered2[filtered2['word'].apply(lambda x: len(wordnet.synsets(x))==0)].copy()

    return filtered3


def filter_archs(df):

    not_words = []

    for i, row in tqdm(df.iterrows()):
        sentence = row['sentence']
        target_word = row['expression'].split()[-1]
        doc = nlp(sentence)

        for token in doc:
            # Check if the token matches the target word and is used as an attribute
            if token.text.lower() == target_word and token.dep_ == "attr":
                # Ensure "is" is the head verb of the target_word
                if token.head.text.lower() == "is":
                    # Extract "X" from the structure "X is [modifiers] [target_word]"
                    is_token_index = token.head.i
                    if is_token_index > 0:
                        x_token = doc[is_token_index - 1]  # Get the token representing "X"
                        x_word = x_token.text
                        
                        # Check for right arcs
                        has_right_arcs = any(child.i > token.i for child in token.children)
                        if not has_right_arcs:
                            # Only add if there are no right arcs
                            not_words.append(i)

    return df.loc[not_words]


def is_word_or_typos(words, spellchecker, threshold=1):
    if spellchecker=='hunspell':
        handler = HunspellHandler()
    elif spellchecker=='spellchecker':
        handler = SpellCheckerHandler()

    word_or_typos = []

    for word in tqdm(words):
        word_typo = True
        if handler.is_misspelled(word):  # not a word (can be a typo or a non-word)
            suggestions = handler.suggest(word)
        
            if suggestions:
                distance = Levenshtein.distance(word, list(suggestions)[0])
                if distance > threshold:  # not a word, not a typo
                    word_typo = False
            else:
                word_typo = False  # not a word, not a typo


        word_or_typos.append(word_typo)

    return word_or_typos


def filter_spellchecker(df, spellchecker_types):
    spellchecker_dir = './data/filtering/spellchecker_removed/'
    os.makedirs(spellchecker_dir, exist_ok=True)
    for sc_type in spellchecker_types:
        print('Using Spellchecker: ', sc_type)
        word_or_typos = is_word_or_typos(df['word'], sc_type, threshold=1)
        df[word_or_typos].to_csv(f'{spellchecker_dir}/{sc_type}.csv')
        df = df[[not w_t for w_t in word_or_typos]].copy()
        print('kept: ', len(df), 'discarded: ', sum(word_or_typos))
    return df



def filter_wiki(df):

    def is_wiki(word):
        searched = wikipedia.search(word)[:3]
        if len(searched)==0:
            return False
        
        for s in searched:
            if word in s.lower():
                return True
        return False
    
    is_wikis = []
    for word in tqdm(df['word']):
        is_wikis.append(is_wiki(word))


    return df[[not i for i in is_wikis]]



def filter_unimorph(df, unimorph_df):
    """
    Filter words based on their presence in the UniMorph dataset.
    Remove any word that exists in the UniMorph lemma list.
    """
    valid_words = set(unimorph_df['lemma']).union(set(unimorph_df['word']))
    filtered_df = df[~df['word'].apply(clean_text).isin(valid_words)].copy()
    
    print(f"Words removed after UniMorph filtering: {len(df) - len(filtered_df)}")
    return filtered_df

def is_valid_word_mw(word, api_key):
    url = f"https://dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and 'meta' in data[0]:
                return True 
    return False 

def filter_merriam_webster(df, api_key):
    valid_words = []
    for word in tqdm(df['word'], desc="Filtering with Merriam-Webster"):
        if is_valid_word_mw(word, api_key):
            valid_words.append(word)

    return df[~df['word'].isin(valid_words)]


def main():
    os.makedirs("./data/csv", exist_ok=True)

    API_KEY = "YOUR_API_KEY"

    # Combine JSON files grouped by expressions into a single CSV file.
    total_csv_path = f"./data/csv/total.csv"
    if not os.path.isfile(total_csv_path):
        total = []
        expressions = ['is a made up word', 'is a made-up word', 'is a created word', 
                       'is not a common word', 'is an invented word', 
                       'is not a real word', 'is not a term', 'is not a word']
        for expression in expressions:
            json_path = f"./data/json/{'_'.join(expression.split())}.json"
            csv_path = f"./data/csv/{'_'.join(expression.split())}.csv"

            with open(json_path, "r") as file:
                data = json.load(file)
                print(f"Data count for '{expression}' : \t{len(data)}")

            df = extract_data(data, csv_path, query=expression)
            df['expression'] = [expression] * len(df)
            total.append(df)

        total_df = pd.concat(total, ignore_index=True)
        print('raw data total', len(total_df))
        total_df = total_df.dropna().drop_duplicates()
        total_df.to_csv(total_csv_path, index=False)
        print(f"Total data saved to {total_csv_path} of length {len(total_df)}")
    else:
        print(f"Loading existing total CSV from {total_csv_path}")
        total_df = pd.read_csv(total_csv_path).dropna().reset_index(drop=True)

    # Word filtering
    df = total_df.copy()
    df1 = filter_words(df)
    save_filtered_and_removed(df, df1, "words")

    df2 = filter_archs(df1)
    save_filtered_and_removed(df1, df2, "archs")

    df3 = filter_spellchecker(df2, ['spellchecker'])
    save_filtered_and_removed(df2, df3, "spell_checker")

    df4 = filter_wiki(df3)
    save_filtered_and_removed(df3, df4, "wiki")

    df5 = filter_merriam_webster(df4, API_KEY)
    save_filtered_and_removed(df4, df5, "merriam_webster")

    os.makedirs("./data/words", exist_ok=True)
    df5.to_csv(f"./data/words/total_non_words.csv", index=False)

    df_dedup = df5.drop_duplicates(subset='word', keep='first')
    df_dedup[['word', 'sentence']].to_csv(f"./data/words/total_non_words_dedup.csv", index=False)
    print(f"Final word list : len {len(df5)} \n Dedupliated word list : len {len(df_dedup)}")


if __name__ == "__main__":
    main()