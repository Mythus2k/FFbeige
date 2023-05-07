import torch, torchtext
from torchtext.functional import to_tensor
from torchtext.data import get_tokenizer

from report_pull import get_report,clean_report
import yfinance as yf 
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import pickle as pkl
from pandas import read_csv

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = get_tokenizer("basic_english")
xlmr_base = torchtext.models.XLMR_BASE_ENCODER
model = xlmr_base.get_model()
model.to(DEVICE)
transform = xlmr_base.transform()

def encode_text(text):
    tokens = tokenizer(text)
    model_input = to_tensor(transform(tokens), padding_value=1)
    return model(model_input)

def pad_tensor(tensor, size=[300,9,768]):
    assert tensor.dim() == 3, "Tensor must be 3D"
    # assert all(s <= size for s in tensor.shape), "Size must be greater than or equal to the tensor shape in all dimensions"
    
    padded_tensor = torch.nn.functional.pad(
        tensor,
        pad=(0, size[-1] - tensor.shape[-1], 0, size[-2] - tensor.shape[-2], 0, size[-3] - tensor.shape[-3]),
        mode="constant",
        value=0,
    )
    
    return padded_tensor

def data_pipeline(train_size=30):
    """
    Returns a training and testing data set from the reports folder
    """
    file = './reports/data.csv'
    data = read_csv(file)
    data = data.sample(len(data))

    train_ref = data.iloc[:train_size]
    test_ref = data.iloc[train_size:]

    train = list()
    test = list()

    for i, row in train_ref.iterrows():
        tensor_name = row['tensor']
        target = torch.tensor([[float(row['target'])]])

        train.append((torch.load(f"./reports/tensors/{tensor_name}.tns"),target))

    for i, row in test_ref.iterrows():
        tensor_name = row['tensor']
        target = torch.tensor([[float(row['target'])]])

        test.append((torch.load(f"./reports/tensors/{tensor_name}.tns"),target))

    return train, test
    
def pad_reports_temp():
    datafile = './reports/data.csv'
    data = open(datafile,'rt')

    data = data.readlines()

    for row in data[1:]:
        row = row.split(',')
        tensor_name = row[0]
        target = row[1]
        tensor = torch.load(f'./reports/tensors/{tensor_name}.tns')
        tensor = pad_tensor(tensor,[300,9,768])
        torch.save(tensor, f'./reports/tensors/{tensor_name}.tns')



if __name__ == '__main__':
    train, test = data_pipeline()

    print(len(train))
    print(train[0])

    print(len(test))
    print(test[0])