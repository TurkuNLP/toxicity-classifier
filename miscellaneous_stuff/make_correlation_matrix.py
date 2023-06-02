import json
import pandas as pd
import sys


"""Small script for making the correlation matrix figure for the toxicity paper."""

data = sys.argv[1]

with open(data, 'r') as json_file:
    json_list = list(json_file)
    lines = [json.loads(jline) for jline in json_list]
# there is now a list of dictionaries
df=pd.DataFrame(lines)

# rename columns
df.rename(columns={'label_toxicity': 'toxicity', 'label_severe_toxicity': 'severe_toxicity', 'label_threat': 'threat', 'label_identity_attack': 'identity_attack', 'label_insult': 'insult', 'label_obscene': 'obscene'}, inplace=True)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation between different variables
corr = df.corr()

# set font scale
#sns.set(font_scale=1.2)

# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(12, 10))

# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))

cmap = sns.color_palette("rocket_r", as_cmap=True) 

heatmap = sns.heatmap(corr, annot=True, annot_kws={ 'fontsize': 16, 'weight':'bold'}, mask = mask, cmap=cmap) 

heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 13, weight='bold')
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 13, weight='bold')

# adjust colorbar font
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)

# save heatmap as an image
fig = heatmap.get_figure()
fig.savefig("corr_matrix_test_2.png") 