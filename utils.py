import pandas as pd
import os

def clean_text(text):
    """lowercase, only english and numbers """
    if isinstance(text, str):
        return re.sub(r'[^a-z0-9]', '', text.lower())
    else:
        return "" 

derivations_file = "./data/unimorph/eng/eng.derivations.tsv"
segmentations_file = "./data/unimorph/eng/eng.segmentations"
args_file = "./data/unimorph/eng/eng.args"

def load_derivations(file_path):
    """Load derivational relationships from eng.derivations.tsv."""
    df = pd.read_csv(file_path, sep="\t", header=None, names=["base", "derived", "relation"])
    df["base"] = df["base"].apply(clean_text)
    df["derived"] = df["derived"].apply(clean_text)
    return df

def load_segmentations(file_path):
    """Load word segmentations from eng.segmentations."""
    segmentations = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                word = clean_text(parts[0]) 
                segmentation = clean_text(parts[1])
                segmentations[word] = segmentation
    return segmentations

def load_args(file_path):
    """Load additional metadata or arguments from eng.args."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return clean_text(content)


def save_filtered_and_removed(original_df, filtered_df, stage_name):
    filtering_dir = './data/filtering/'    
    os.makedirs(filtering_dir, exist_ok=True)

    removed_df = original_df[~original_df.index.isin(filtered_df.index)]
    
    filtered_df.to_csv(f"{filtering_dir}/{stage_name}_filtered.csv", index=False)    
    removed_df.to_csv(f"{filtering_dir}/{stage_name}_removed.csv", index=False)
    
    print(f"{stage_name}: Filtered {len(filtered_df)} words, Removed {len(removed_df)} words.")



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
