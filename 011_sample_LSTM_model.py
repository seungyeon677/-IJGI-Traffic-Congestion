import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

import optuna

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# data scaling
def data_scaling(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    max = data.max()
    min = data.min()

    return scaler, scaled_data, min, max

# create target, y tensor; target = t, y = t+7
def create_weekly_sequence_tensor(scaled_data, seq=7):
    X, y = [], []
    for i in range(365-seq):
        X.append([scaled_data[i: i+seq]])
        y.append(scaled_data[i+seq])
    
    return torch.FloatTensor(X).to(device), torch.FloatTensor(y).to(device).view([-1, 1])

def create_dataset(data_X, data_y, train = 275, validation = 299, batch_size = 20):
    # train, test data split
    X_train, X_val, X_test = data_X[:train], data_X[train:validation], data_X[validation:]
    y_train, y_val, y_test = data_y[:train], data_y[train:validation], data_y[validation:]

    # create tensorDataset based on train, test data
    train = TensorDataset(X_train, y_train)
    val = TensorDataset(X_val, y_val)
    test = TensorDataset(X_test, y_test)

    # create batch data set
    train_loader = DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test, batch_size = 59, shuffle = False)
    val_loader = DataLoader(dataset = val, batch_size = batch_size, shuffle = True)

    return train_loader, val_loader, test_loader

# define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq, device):
        super(LSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*seq, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        fc_out = self.fc(lstm_out.contiguous().view(x.size(0), -1))

        return lstm_out, hn, fc_out

# train LSTM model
def LSTM_train(model, train_loader, val_loader, criterion, optimizer):
    model.train()
    total_train_loss = 0
    for t, data in enumerate(train_loader):
        seq, target = data
        out, hidden, fully = model(seq)
        
        loss = criterion(fully, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    ave_train_loss = total_train_loss / (t+1)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for v, val_data in enumerate(val_loader):
            val_seq, val_target = val_data
            val_out, val_hidden, val_fully = model(val_seq)

            loss = criterion(val_fully, val_target)
            total_val_loss += loss.item()

        ave_val_loss = total_val_loss / (v+1)

    return ave_train_loss, ave_val_loss

# test LSTM model
def LSTM_test(model, test_loader, criterion):
    # evaluate model
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for t, data in enumerate(test_loader):
            seq, target = data
            out, hidden, fully = model(seq)

            loss = criterion(fully, target)
            total_loss += loss.item()

        ave_loss = total_loss / (t+1)

        return fully.cpu().numpy(), target.cpu().numpy(), ave_loss


# find best hyperparameter set
def objective(trial):
    # Hyperparameter search space
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    epochs = trial.suggest_categorical("epoch", [10, 30, 50])
    
    # Initialize model, optimizer, and loss function
    model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                 seq=1, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        _, ave_val_loss = LSTM_train(model, train_loader, val_loader, criterion, optimizer)
    
    return ave_val_loss

"""==============================================================================================="""

df = pd.read_csv('./sample_dataset.csv')
df.head()

day_col = [col for col in df.columns if col.startswith('Day')]

scaler_list, cities_scaled, min_list, max_list = [], [], [], []
for i in range(len(df)):
    scaler, data, min, max = data_scaling(df.loc[i, day_col].values.reshape(-1, 1))
    cities_scaled.append(data.reshape(1, -1))
    scaler_list.append(scaler)
    min_list.append(min)
    max_list.append(max)

city_X, city_y = [], []
for city in tqdm(cities_scaled):
    X, y = create_weekly_sequence_tensor(city[0])
    city_X.append(X)
    city_y.append(y)


hp = []
for i in range(len(df)):
    # create train, validation, test dataset
    train_loader, val_loader, test_loader = create_dataset(city_X[i], city_y[i])
    test_index = np.array(range(307, 366))
    print("Finish creating datasets")
    
    # base parameter
    input_size = 7      # 7 days
    seq = 1             # 1 week
    lr = 0.001          # learning rate

    # Run Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Adjust n_trials as needed for more thorough search

    # hyperparameter tunning
    hidden_size = study.best_params['hidden_size']      
    num_layers = study.best_params['num_layers']        
    epochs = study.best_params['epoch']                 

    # create LSTM mdoel
    model = LSTM(input_size = input_size, hidden_size = hidden_size,
                 num_layers = num_layers, seq = seq, device = device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    print('Finish creating LSTM model')

    # train model
    train_loss_graph = []
    val_loss_graph = []
    
    for epoch in range(epochs):
        ave_train_loss, ave_val_loss = LSTM_train(model, train_loader, val_loader, criterion, optimizer)
        train_loss_graph.append(ave_train_loss)
        val_loss_graph.append(ave_val_loss)
        print(f"Epoch {epoch+1} - Train MSE: {ave_train_loss:.4f} | Val MSE: {ave_val_loss:.4f}") 

    ### make loss graph
    plt.figure(figsize=(16,8))
    plt.plot(train_loss_graph, c = 'red', label = f'Train Loss (RMSE: {np.sqrt(ave_train_loss):.4f})')
    plt.plot(val_loss_graph, c = 'blue', label = f'Validation Loss (RMSE: {np.sqrt(ave_val_loss):.4f})')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.ylabel('Loss', fontsize = 18)
    plt.xlabel('Epoch', fontsize = 18)
    plt.title(df.loc[i, 'Name'], y=1.0, pad=-25, weight = 'bold', fontsize = 23)
    plt.legend(loc = 'upper right', fontsize = 15)
    plt.savefig(f"./LSTM_train_result_{df.loc[i, 'Name']}.png", dpi = 500)
    # plt.show() 


    # evaluate model
    pred, act, ave_loss = LSTM_test(model, test_loader, criterion)
    rmse = np.sqrt(ave_loss)
    print(f"Test MSE: {ave_loss: .4f}")


    pred_rescale = scaler_list[i].inverse_transform(pred)
    act_rescale = scaler_list[i].inverse_transform(act)
    rmse_rescale = np.sqrt(mean_squared_error(act_rescale, pred_rescale))


    # 실제 값과 예측 값 시각화
    plt.figure(figsize=(24, 6))
    plt.plot(test_index, act_rescale, label='Observed value', color='black', linestyle='--', marker = 'o')
    plt.plot(test_index, pred_rescale, label='Predicted value', color='red', linestyle='-', marker = 'o')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Day', fontsize = 18)
    plt.ylabel('Congestion (mileDay)', fontsize = 18)
    plt.title(f"{df.loc[i, 'Name']}; RMSE (scaled: {rmse:.4f}, unscaled: {rmse_rescale:.4f})", y=1.0, pad=-25, weight = 'bold', fontsize = 23)
    plt.legend(loc = 'upper left', fontsize = 15)
    plt.savefig(f"./LSTM_test_result_{df.loc[i, 'Name']}.png", dpi = 500)
    # plt.show()

    hp.append([df.loc[i, 'Name'], hidden_size, num_layers, lr, epochs, val_loss_graph[-1], rmse])
    print(hp)

hp_df = pd.DataFrame(hp, 
                     columns = ['Name', 'Hidden dim', 'Hidden layer', 'Learning rate', 'Epochs', 'val_loss (MSE)', 'test_loss (RMSE)'])
hp_df.to_csv('./cities_test.csv', index = False)