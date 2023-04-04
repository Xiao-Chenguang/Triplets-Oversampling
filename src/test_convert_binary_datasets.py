import pandas as pd
from to_binary_csv import convert_to_csv

convert_to_csv()
print('Converrt Done')
datasets = ['abalone', 'diabete', 'ionosphere', 'satimage', 'vowel', 'aloi', 'haberman', 'pulsar', 'vehicle']

for dataset in datasets:
    print(f'==================== {dataset} ====================')
    df = pd.read_csv(f'datasets/{dataset}.csv', header=None)
    print(f'{dataset}: {df.shape}')
    cnt = df.iloc[:,-1].value_counts()
    print(cnt)
    print(f'{cnt.keys()[0]} / {cnt.keys()[1]} = {cnt[0] / cnt[1]}')
    print('=' * 50)
