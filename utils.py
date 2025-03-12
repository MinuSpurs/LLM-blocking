import pandas as pd
import os
import re

def clean_text(text):
    """lowercase, only english and numbers """
    if isinstance(text, str):
        return re.sub(r'[^a-z]', '', text.lower())
    else:
        return "" 

def strip_special_chars_edges(word):
    if isinstance(word, str):
        return re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
    return word


def save_filtered_and_removed(original_df, filtered_df, stage_name, filtering_dir='./data/filtering/'):
    os.makedirs(filtering_dir, exist_ok=True)

    removed_df = original_df[~original_df.index.isin(filtered_df.index)]
    
    filtered_df.to_csv(f"{filtering_dir}/{stage_name}_filtered.csv", index=False)    
    removed_df.to_csv(f"{filtering_dir}/{stage_name}_removed.csv", index=False)
    
    print(f"{stage_name}: Filtered {len(filtered_df)} words, Removed {len(removed_df)} words.")


def load_unimorph_dataset(file_path):
    """Load UniMorph dataset (eng.txt) into a DataFrame."""
    df = pd.read_csv(file_path, sep="\t", header=None, names=["word", "lemma", "features"])
    df["word"] = df["word"].apply(clean_text)
    df["lemma"] = df["lemma"].apply(clean_text)
    df = df.dropna()
    return df


class HunspellHandler:
    def __init__(self):
        from hunspell import Hunspell
        self.hspell = Hunspell('en_US')

    def is_misspelled(self, word):
        return not self.hspell.spell(word)

    def suggest(self, word):
        return self.hspell.suggest(word)


class SpellCheckerHandler:
    def __init__(self):
        from spellchecker import SpellChecker
        self.spell = SpellChecker()

    def is_misspelled(self, word):
        return not self.spell.known([word])

    def suggest(self, word):
        return self.spell.candidates(word)
