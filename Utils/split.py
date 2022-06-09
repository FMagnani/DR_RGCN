import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.9          # The portion of triplets used for training
TEST_TO_VALID_SIZE = 0.5  # The ratio between test and validation sets

### Load whole dataset
load_path = "drkg.tsv"
df = pd.read_csv(load_path, sep='\t', header=None, names=['Source','Rel','Target'])

### Split
trainDF, testDF = train_test_split(df, train_size=TRAIN_SIZE, random_state=53)
testDF, validDF = train_test_split(testDF, test_size=TEST_TO_VALID_SIZE, random_state=53)

### Log the number of triplets of each split
total_edges = df.shape[0]
train_edges = trainDF.shape[0]
test_edges = testDF.shape[0]
valid_edges = validDF.shape[0]
print("Total edges: ", total_edges, "\n",
      "Train edges: ", train_edges, " (", train_edges/total_edges*100, "%)\n",
      "Test edges: ", test_edges, " (", test_edges/total_edges*100, "%)\n",
      "Valid edges: ", valid_edges, " (", valid_edges/total_edges*100, "%)\n"
     )

### Save in the current directory
# In tsv format but txt extension
# See also https://aws-dglke.readthedocs.io/en/latest/train.html
trainDF.to_csv("train.txt", header = None, index = None, sep = "\t")
testDF.to_csv("test.txt", header = None, index = None, sep = "\t")
validDF.to_csv("valid.txt", header = None, index = None, sep = "\t")
