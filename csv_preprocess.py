import spacy
import pandas as pd
from tqdm import tqdm
from utils import *
from personal import DICTIONARY_KEY, MEDICAL_KEY
import Levenshtein
from nltk.corpus import words, wordnet
import nltk
import os
import re 
import requests
import wikipediaapi
import wikipedia


# nltk.download('words')
# nltk.download('wordnet')   # Remove comments as needed
nlp = spacy.load("en_core_web_sm")


def process_word(word):
    word = word.lower()

    # Count hyphens
    hyphen_count = word.count('-')

    # No hyphen OR a single hyphen (which is removed)
    if hyphen_count <= 1:
        cleaned_word = word.replace('-', '') if hyphen_count == 1 else word
        return cleaned_word if re.fullmatch(r'^[a-zA-Z]+$', cleaned_word) else False

    # Two or more hyphens → Invalid
    return False

# csv filtering - only EN, 23>length>3, 3> consecutive letters, not nltk word, wordnet
def filter_words(df):
    english_words = set(words.words())
    consecutive_pattern = r'(.)\1{2,}' 

    filtered1 = df[df['word'].str.len().between(4, 22)]
    filtered2 = filtered1[filtered1['word'].apply(lambda x: not bool(re.search(consecutive_pattern, x)))]
    filtered2 = filtered2[~filtered2['word'].isin(english_words)].copy()
    filtered3 = filtered2[filtered2['word'].apply(lambda x: len(wordnet.synsets(x))==0)].copy()

    return filtered3


def filter_archs(df):

    not_words = []

    for i, row in tqdm(df.iterrows()):
        sentence = row['sentence']
        target_word = row['query'].split()[-1]
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


def filter_spellchecker(df, spellchecker_types, filtering_path):
    spellchecker_dir = f'{filtering_path}/spellchecker_removed/'
    os.makedirs(spellchecker_dir, exist_ok=True)
    for sc_type in spellchecker_types:
        print('Using Spellchecker: ', sc_type)
        word_or_typos = is_word_or_typos(df['word'], sc_type, threshold=1)
        df[word_or_typos].to_csv(f'{spellchecker_dir}/{sc_type}.csv')
        df = df[[not w_t for w_t in word_or_typos]].copy()
        print('kept: ', len(df), 'discarded: ', sum(word_or_typos))
    return df


def filter_wiki(df):

    def is_wiki(word, wiki):

        searched = wikipedia.search(word)[:3]
        if len(searched)==0:
            return False
        
        for s in searched:
            if Levenshtein.distance(word, s.lower())<2:
                return True
        try:  
            page_by = wiki.page(word)  
            if page_by.fullurl != -1:
                return True
        except:
            return False
        return False
    
    wiki = wikipediaapi.Wikipedia(user_agent='changeling', language='en')
    is_wikis = []
    for word in tqdm(df['word']):
        is_wikis.append(is_wiki(word, wiki))

    return df[[not i for i in is_wikis]]


def is_valid_word_merriam_webster(word, collegiate_key, medical_key):
    """
    Check if a word exists in either the Collegiate API or Medical API.
    Stops checking after finding the word in one API.
    """
    base_urls = {
        "collegiate": f"https://dictionaryapi.com/api/v3/references/collegiate/json/{word}?key={collegiate_key}",
        "medical": f"https://www.dictionaryapi.com/api/v3/references/medical/json/{word}?key={medical_key}",
    }

    for api_name, url in base_urls.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict) and 'meta' in data[0]:
                            return True
                except Exception as e:
                    print(f"Error decoding JSON for {api_name.capitalize()} API: {e}")
            else:
                print(f"Non-200 response from {api_name.capitalize()} API: {response.status_code}")
        except Exception as e:
            print(f"Request failed for {api_name.capitalize()} API: {e}")
    return False


def filter_merriam_webster(df, collegiate_key, medical_key):
    """
    Filter words using Merriam-Webster's Collegiate and Medical APIs.
    """
    valid_words = []
    for word in tqdm(df['word']):
        if is_valid_word_merriam_webster(word, collegiate_key, medical_key):
            valid_words.append(word)

    return df[~df['word'].isin(valid_words)]


def main():

    total_csv_path = './data/csv/total.csv'
    filtering_path = './data/filtering'    
    word_path = './data/words'

    total_df = pd.read_csv(total_csv_path) # debug : .iloc[:100]

    # Set total.csv file to be preprocessed
    total_df['org'] = total_df['word']
    total_df['word'] = total_df['word'].astype(str).apply(process_word)
    total_df = total_df[total_df['word'] != False].dropna().reset_index(drop=True)
    total_df.to_csv(total_csv_path.split('.csv')[0] + '_preprocess.csv', index=False)
    print(f"Total data {len(total_df)} saved to {total_csv_path.split('.csv')[0] + '_.csv'}")

    # Word filtering
    df = total_df.copy()
    df1 = filter_words(df)
    save_filtered_and_removed(df, df1, "words", filtering_path)

    df2 = filter_archs(df1)
    save_filtered_and_removed(df1, df2, "archs", filtering_path)

    df3 = filter_spellchecker(df2, ['spellchecker'], filtering_path)
    save_filtered_and_removed(df2, df3, "spell_checker", filtering_path)

    df4 = filter_wiki(df3)
    save_filtered_and_removed(df3, df4, "wiki", filtering_path)

    df5 = filter_merriam_webster(df4, DICTIONARY_KEY, MEDICAL_KEY)
    save_filtered_and_removed(df4, df5, "merriam_webster", filtering_path)

    os.makedirs(word_path, exist_ok=True)
    df5.to_csv(f"{word_path}/total_non_words.csv", index=False)

    df_dedup = df5.drop_duplicates(subset='word', keep='first')
    df_dedup[['word', 'sentence']].to_csv(f"{word_path}/total_non_words_dedup.csv", index=False)
    print(f"Final word list : len {len(df5)} \n Dedupliated word list : len {len(df_dedup)}")


if __name__ == "__main__":
    main()
