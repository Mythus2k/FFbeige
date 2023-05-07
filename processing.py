import torch, torchtext
from torchtext.functional import to_tensor
from torchtext.data import get_tokenizer

from report_pull import get_report,clean_report
import yfinance as yf 
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas import read_csv, concat, DataFrame

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

def add_report(month=int(),year=int(),header=str(),paragraph=str()):
    assert month < 12 and month > 0, "Month must be between 1 and 12"
    assert year > 1996, "Beige report archive only goes back to 1996"
    print(f'Processing: Adding Report - R{month}{year}{header[:5]}{paragraph[:5]}')

    report = get_report(str(month),str(year))
    if report == None:
        print("Unable to get report")
        return None

    report = clean_report(report)

    if header not in report.keys():
        print(f'Header not in report - try again with one of these:\n {report.keys()}')
        return None
    section = report[header]

    if paragraph not in section.keys():
        print(f"Paragraph not in report - try again with one of these:\n {section.keys()}")
        return None
    text = section[paragraph]

    token = encode_text(text)
    token = pad_tensor(token)
    tensor_name = f"R{month}{year}{header[:5]}{paragraph[:5]}"

    start_date = dt(year,month,1)
    end_date = start_date + relativedelta(months=2)
    if end_date > dt.now(): print(f" Warning: performance target does not contain full 2 months\n end_date: {end_date}, today: {dt.now()}")

    spy = yf.download('SPY',start=start_date.strftime('%Y-%m-%d'),end=end_date.strftime('%Y-%m-%d'))['Adj Close'].pct_change()

    if spy.sum() > 0: target = 1
    else: target = 0

    data = open('./reports/data.csv','at')
    data.write(f"{tensor_name},{target}\n")
    data.close()
    torch.save(token,f'./reports/tensors/{tensor_name}.tns')

    print(' Completed Successfully')
    return None



if __name__ == '__main__':

    for year in range(2017, 2022):
        for month in range(1,12):
            add_report(month, year, 'Overall Market', 'Employment and Wages')