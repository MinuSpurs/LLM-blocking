import json
import spacy
import pandas as pd
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


# extracting a word and a sentence
def extract_data(data, output_file, query="is a made up word"):
    
    ixs, words, texts = [], [], []

    for document in tqdm(data):
        spans = document.get("spans", [])
        ix = document.get("doc_ix", [])

        word = spans[0][0].split()[-1]
        text = " ".join([span[0] for span in spans])

        if word != [] and text != '':
            words.append(word)
            texts.append(text)
            ixs.append(ix)

    docs = nlp.pipe(texts, batch_size=100)  
    
    sentences_with_query = [] 

    for doc in docs:
        for sentence in doc.sents:
            if query in sentence.text:

                sentences_with_query.append(sentence.text.strip())
                break

    # Save csv file

    output_file = f"{output_file.split('.json')[0]}_sentence.csv"

    df = pd.DataFrame()
    df['doc_ix'] = ixs
    df['word'] = words
    df['sentence'] = sentences_with_query
    df.to_csv(output_file, index=False)

    return df


# nltk filtering
def filter_wordnet():
    pass

def filter_archs(sentences):

    not_words = []
    
    for sentence in sentences:
        doc = nlp(sentence)
        for token in doc:


            # Identify the "word" token in the structure "X is not a word"
            if token.text.lower() == "word" and token.dep_ == "attr":   # 문장이 바뀔때 수정해야함
                # Check if "is" exists as an auxiliary verb right before "not"
                if token.head.text.lower() == "is" and any(child.text.lower() == "not" for child in token.head.children):
                    # Identify "X" as the token immediately before "is"
                    is_token_index = token.head.i
                    if is_token_index > 0:
                        x_token = doc[is_token_index - 1]
                        x_word = x_token.text

                        has_right_arcs = any(child.i > token.i for child in token.children)

                        # Only apply basic checks for WordNet existence and stop words
                        if not has_right_arcs and not is_real_word(x_word, x_token.tag_):
                            print(f"Adding '{x_word}' to not_words list.")
                            not_words.append((x_word, sentence))  # Save the word along with the original sentence
                        else:
                            print(f"Skipping '{x_word}' (exists in WordNet, is a common word/stop word, or is a proper noun).")
    
    return not_words

def main():

    # expressions = ['is a made-up word', 'is a made up word', 'is not a common word', 'is a created word', 'is an invented word']
    expressions = ['is a made up word', 'is not a common word', 'is a created word', 'is an invented word']
    
    for expression in expressions:

        file_path = f"./data/finished/{'_'.join(expression.split())}.json"
        
        with open(file_path, "r") as file:
            data = json.load(file)
            print(f"Data count for '{expression}' : \t{len(data)}")
        
        df = extract_data(data, file_path, query=expression)
        # words = extract_not_words_from_sentences(sentences)

main()

