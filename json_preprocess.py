import json
import spacy
import pandas as pd
from tqdm import tqdm
from hunspell import HunSpell
import Levenshtein
from nltk.corpus import words, wordnet
import nltk

#nltk.download('words')
#nltk.download('wordnet')   # Remove comments as needed

# Hunspell setup
h = HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')

with open('/usr/share/hunspell/en_US.dic', 'r') as file:
    dictionary = [line.strip() for line in file.readlines() if line.strip()]


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

def save_filtered_and_removed(original_df, filtered_df, stage_name):    # 필터링에서 조건에 부합하지 않은(탈락한) 단어들 저장하는 함수

    removed_df = original_df[~original_df.index.isin(filtered_df.index)]
    
    filtered_df.to_csv(f"./data/csv/{stage_name}_filtered.csv", index=False)    # 각 단계별로 저장되고 탈락된 단어들이 모아짐
    removed_df.to_csv(f"./data/csv/{stage_name}_removed.csv", index=False)
    
    print(f"{stage_name}: Filtered saved to {stage_name}_filtered.csv, Removed saved to {stage_name}_removed.csv")



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

                        # Append the word, sentence, and doc_ix to the result
                        # print(f"Adding '{x_word}' to not_words list.")  # Handle comment if necessary
                        not_words.append((x_word, sentence, doc_ix)) 

    filtered_df = pd.DataFrame(not_words, columns=['word', 'sentence', 'doc_ix'])
    return filtered_df

def filter_hunspell(df):
    filtered_words = []
    for _, row in df.iterrows():
        word = row['word']
        if not h.spell(word):  # If the word is not in the Hunspell dictionary
            filtered_words.append(row)

    filtered_df = pd.DataFrame(filtered_words, columns=['word', 'sentence', 'doc_ix'])
    return filtered_df


def filter_levenshtein(df, dictionary, threshold=2):
    filtered_words = []
    for _, row in df.iterrows():
        word = row['word']
        closest_word = min(dictionary, key=lambda x: Levenshtein.distance(word, x))
        distance = Levenshtein.distance(word, closest_word)
        if distance > threshold:
            filtered_words.append(row)

    filtered_df = pd.DataFrame(filtered_words, columns=['word', 'sentence', 'doc_ix'])
    return filtered_df


def main():
    """
    Main function to process JSON data and filter words.
    """
    total = []
    expressions = ['is a created word']  # Add more expressions as needed

    for expression in expressions:
        target_word = expression.split()[-1]

        json_path = "/home/work/jupyter/minwoo/CMU/LLM_blocking/data/json/is_a_created_word.json"
        csv_path = f"./data/csv/{'_'.join(expression.split())}.csv"

        with open(json_path, "r") as file:
            data = json.load(file)
            print(f"Data count for '{expression}' : \t{len(data)}")

        df = extract_data(data, csv_path, query=expression)

        df1 = filter_words(df)
        save_filtered_and_removed(df, df1, "filter_words")
        
        df2 = filter_archs(df1, target_word)
        save_filtered_and_removed(df1, df2, "filter_archs")
        
        df3 = filter_hunspell(df2)
        save_filtered_and_removed(df2, df3, "filter_hunspell")
        
        df4 = filter_levenshtein(df3, dictionary)
        save_filtered_and_removed(df3, df4, "filter_levenshtein")

        df4.to_csv(f"{csv_path.split('.csv')[0]}_filtered.csv", index=False)
        print(f"Saving filtered CSV with len {len(df4)}")

        total.append(df4)

    total_df = pd.concat(total, ignore_index=True)
    csv_path = f"./data/words/total_non_words.csv"
    total_df.to_csv(csv_path, index=False)
    print(f"Total filtered words saved to {csv_path}")


main()
