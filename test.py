from tqdm.auto import tqdm 
from torch.utils.data import DataLoader
import pandas as pd

try:
    from .data_preprocess import STRDataset
except ImportError:
    from data_preprocess import STRDataset


root_dir = './IAM/image/'
df = pd.read_fwf('./IAM/gt_test.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
del df[2]
df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)
dataset = STRDataset(root_dir=root_dir, df=df)
datas = DataLoader(dataset=dataset, batch_size=1)

images = []
texts = []
for i, data in enumerate(tqdm(datas)):
    img, label = data
    print(img, label)
    if i == 2:
        break
