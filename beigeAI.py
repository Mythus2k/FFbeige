import torch
from torch import nn, optim

from processing import data_pipeline, encode_text, pad_tensor
from report_pull import get_report, clean_report

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class biege(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(biege, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(64 * 600, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv1(x) 
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x) 
        x = torch.relu(x)
        x = self.pool2(x) 
        x = x.view(-1, 64 * 600)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    

model = biege(input_dim=768, output_dim=1)
model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


# train_data, test_data = data_pipeline(90)


num_epochs = 20

for epoch in range(num_epochs):
    train_loss = 0.0
    train_data, test_data = data_pipeline(90)
    for i, (inputs, labels) in enumerate(train_data):
        # move data to device
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass and optimize
        loss.backward()
        optimizer.step()

        # update running loss
        train_loss += loss.item()

    # calculate average training loss
    avg_train_loss = train_loss / len(train_data)

    # evaluate on test set
    with torch.no_grad():
        test_loss = 0.0
        correct_count = 0
        pred_buy = 0
        for inputs, labels in test_data:
            # move data to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # update running loss
            test_loss += loss.item()
            pred = (outputs > 0.5).float() # convert to binary predictions
            correct_count += (pred == labels).sum().item()
            pred_buy += (1. == pred).sum().item()
        # calculate average test loss
        avg_test_loss = test_loss / len(test_data)
        accuracy = correct_count / len(test_data)
        pred_buy_amt = pred_buy / len(test_data)


    # print epoch statistics
    print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.4f}, test_loss={avg_test_loss:.4f}, accuracy: {accuracy:.6f}, predicted buy: {pred_buy_amt:.6}, len_data: {len(test_data)}")

torch.save(model,'./models/biege_0.05.model')


def perfomance_check(month,year):
    print(f"Checking {month}/{year}")
    report = get_report(month,year)
    report = clean_report(report)
    section = report['Overall Market']['Overall Economic Activity']
    tensor = encode_text(section)
    tensor = pad_tensor(tensor)

    out = model(tensor)
    print(out)


print(f"Performance Check")
perfomance_check('01','2023')
perfomance_check('03','2023')
perfomance_check('04','2023')
