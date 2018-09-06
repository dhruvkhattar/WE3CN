import pickle as pkl
import numpy as np
import json
import pdb
from tqdm import tqdm

K = 50

article_info = json.load(open('../data/articles.json'))
articles = article_info.keys()
embed = json.load(open('../data/glove_embed.json'))

mat = {}

for art in tqdm(articles):
    text = article_info[art]['title'] + article_info[art]['text']
    temp = []
    for word in text[:K]:
        if word.lower() in embed:
            temp.append(embed[word.lower()])
        else:
            temp.append([0]*300)
    for i in range(K-len(text)):
            temp.append([0]*300)

    mat[art] = np.asarray(temp)

pkl.dump(mat, open('../data/mat.pkl', 'w'))
