import pandas as pd

df = pd.read_csv(r'list_attr_celeba.csv')
df = df.set_index('image_id')
df = df.transpose()

df = df.replace(to_replace=1, value="male")
df = df.replace(to_replace=-1, value="female")
df.to_json (r'CelebA.json')