import pyarrow.parquet as pq
import pandas as pd


parquet_file = pq.ParquetFile("train_ember_2018_v2_features.parquet")


label_col = "Label"  

malware_chunks = []
benign_chunks = []


for batch in parquet_file.iter_batches(batch_size=5000):
    df = batch.to_pandas()
    malware_chunks.append(df[df[label_col] == 1])
    benign_chunks.append(df[df[label_col] == 0])
    
   
    if sum(len(x) for x in malware_chunks) >= 2500 and \
       sum(len(x) for x in benign_chunks) >= 2500:
        break

malware = pd.concat(malware_chunks).head(2500)
benign = pd.concat(benign_chunks).head(2500)

subset = pd.concat([malware, benign])
subset.to_csv("ember_subset.csv", index=False)

print("subset créé avec succès")
