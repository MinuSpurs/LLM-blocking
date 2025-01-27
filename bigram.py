import csv
import subprocess

csv_file_path = "./data/csv/total.csv"
output_file_path = "output_segmented.txt"
bigram_segment_path = "./legros/src/bigram_segment"
bigram_stats_path = "./legros/from_morf-bpe128k/bigram_stats.19"
unigram_stats_path = "./legros/from_morf-bpe128k/unigram_stats.19"

beam_size = 5

words = []
with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        words.append(row['word'])

with open(output_file_path, 'wb') as output_file: 
    for word in words:
        process = subprocess.run(
            [
                bigram_segment_path,
                bigram_stats_path,
                unigram_stats_path,
                "-b", str(beam_size)
            ],
            input=word.encode(),  
            capture_output=True
        )
        output_file.write(process.stdout) 

print(f"The segmentation results have been saved to {output_file_path}.")