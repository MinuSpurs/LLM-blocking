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


# csv filtering - only EN, length>3, 3> consecutive letters, not nltk word, wordnet
def filter_words(df):
    english_words = set(words.words())
    consecutive_pattern = r'(.)\1{2,}' 

    filtered = df[df['word'].str.match('^[a-zA-Z]+$')]
    filtered1 = filtered[filtered['word'].str.len()>3]
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


def filter_spellchecker(df, spellchecker_types):
    print(spellchecker_types)
    spellchecking = SpellChecking(spellchecker_types)
    for spellchecker in spellchecker_types:
        df = df[df['word'].apply(lambda x : spellchecking.is_misspelled(x, spellchecker))]
    return df


def filter_levenshtein(df, dictionary, threshold=2):

    flags = []
    for word in df['word']:
        closest_word = min(dictionary, key=lambda x: Levenshtein.distance(word, x))
        distance = Levenshtein.distance(word, closest_word)

        if distance <= threshold:  
            flags.append(True)
        else:
            flags.append(False)

    return df[flags]

def filter_wiki(df):

    def is_wiki(word):
        searched = wikipedia.search(word)
        if len(searched)==0:
            return False
        
        for s in searched:
            if word in s.lower():
                return True
        return False

    return df[~df['word'].apply(is_wiki)]



dictionary_file = "./data/spelling/en_US.dic" 
with open(dictionary_file, 'r', encoding='latin1') as file:
    dictionary = [line.strip() for line in file.readlines() if line.strip()]  



def load_unimorph_dataset(file_path):
    """Load UniMorph dataset (eng.txt) into a DataFrame."""
    df = pd.read_csv(file_path, sep="\t", header=None, names=["word", "lemma", "features"])
    df["word"] = df["word"].apply(clean_text)
    df["lemma"] = df["lemma"].apply(clean_text)
    df = df.dropna()
    return df

def filter_unimorph(df, unimorph_df):
    """
    Filter words based on their presence in the UniMorph dataset.
    Remove any word that exists in the UniMorph lemma list.
    """
    valid_words = set(unimorph_df['lemma'])
    filtered_df = df[~df['word'].apply(clean_text).isin(valid_words)].copy()
    
    print(f"Words removed after UniMorph filtering: {len(df) - len(filtered_df)}")
    return filtered_df


def main():
    """
    Main function to process JSON data and filter words.
    """

    os.makedirs("./data/csv", exist_ok=True)

    # Combine JSON files grouped by expressions into a single CSV file.
    total_csv_path = f"./data/csv/total.csv"
    if not os.path.isfile(total_csv_path):
        total = []
        expressions = ['is a created word']  # Add more expressions as needed
        # expressions = ['is a made up word', 'is not a common word', 'is a created word', 'is an invented word']
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
        total_df = total_df.dropna()
        total_df = total_df.drop_duplicates() # deleting duplicates for exact same rows
        total_df.to_csv(total_csv_path, index=False)
        print(f"Total data saved to {total_csv_path} of length {len(total_df)}")
    else:
        print(f"Loading existing total CSV from {total_csv_path}")
        total_df = pd.read_csv(total_csv_path)



    # Word filtering
    df = total_df.copy()
    df1 = filter_words(df)
    save_filtered_and_removed(df, df1, "filter_words")
    
    
    df2 = filter_archs(df1)
    save_filtered_and_removed(df1, df2, "filter_archs")
    
    '''df3 = filter_hunspell(df2)
    save_filtered_and_removed(df2, df3, "filter_hunspell")'''

    df3 = filter_spellchecker(df2)
    save_filtered_and_removed(df2, df3, "filter_spellchecker")
    
    # Load dictionary from the en_US.dic file
    with open('./Library/Spelling/en_US.dic', 'r', encoding='latin1') as file:
        dictionary = [line.strip() for line in file.readlines() if line.strip()]

    df4 = filter_levenshtein(df3, dictionary)
    save_filtered_and_removed(df3, df4, "filter_levenshtein")

    df5 = filter_wiki(df4)
    save_filtered_and_removed(df4, df5, "filter_wiki")

    eng_file = "./data/unimorph/eng/eng.txt"
    unimorph_df = load_unimorph_dataset(eng_file)
    df6 = filter_unimorph(df5, unimorph_df)
    save_filtered_and_removed(df5, df6, "filter_unimorph")

    os.makedirs("./data/words", exist_ok=True)
    df6.to_csv(f"./data/words/total_non_words.csv", index=False)

    df_dedup = df6.drop_duplicates(subset='word', keep='first')
    df_dedup[['word', 'sentence']].to_csv(f"./data/words/total_non_words_dedup.csv", index=False)
    print(f"Final word list : len {len(df6)} \n Dedupliated word list : len {len(df_dedup)}")

if __name__ == "__main__":
    main()
