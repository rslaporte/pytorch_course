# %%
#DOWNLOADING THE HELPER FUNCTIONS FROM GITHUB
import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading the helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

#%%
############# 1. NEURAL NETWORK CLASSIFICATION - IDENTIFY THE CIRCLE #############
from helper_functions import plot_predictions, plot_decision_boundary

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

#Creating a circle toy data
X, y = make_circles(n_samples=1000,
                    noise=0.03,
                    random_state=42)

plt.figure(figsize=(10,10))
plt.title("Toy circle data")
plt.scatter(x=X[:,0], 
            y=X[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

#Turning into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

len(X_test), len(X_train), len(y_test), len(y_train)

#Model Building (agnostic code)
device = "cuda" if torch.cuda.is_available() else "cpu"
device

############# MODELING THE CIRCLE WITH LINEAR LAYERS #############
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))

#Using nn.Sequential instead of creating class and overwriting the forward method
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=2)
)

with torch.inference_mode():
    untrained_preds = model_0(X_test)
    print(torch.round(untrained_preds[:10]))

#Loss and Optimize Functions
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#Evaluating
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = 100*correct/len(y_pred)
    return acc

'''
    Training model steps:
    1 - Forward pass
    2 - Calculate the loss
    3 - Optimizer zero grad
    4 - Loss backwards (backpropagation)
    5 - Optimizer sstep (gradient descent)
'''
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))
    y_pred_probs = torch.sigmoid(y_logits)

torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    #Training
    model_0.train()

    #1. Forward Pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #2. Calculate the loss/accuracy
    loss = loss_fn(y_logits, 
                   y_train)    
    
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. Loss backward (backpropagaton)
    loss.backward()

    #5. Optimizer step (gradient descent)
    optimizer.step()

    #Testing
    model_0.eval()
    with torch.inference_mode():
        #1 Forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        #2 Calculate the test loss/acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, 
                               y_pred=test_pred)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}")

############# Ploting the predictions data, checking the model's classificatio
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Training")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)

#%% 
############## 2. USING THE NEURAL NETWORK TO CLASSIFY A LINEAR PROBLEM (CHECKING IF OUR MODEL CAN LEARN) ############## 

#Creating a linear toy data
X = torch.arange(0,1,0.02).unsqueeze(dim=1) 
y = 0.7*X + 0.3 #y = weight * X + bias

train_split = int(0.8*len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

#Creating the model
model_1 = nn.Sequential(
    nn.Linear(in_features=1, out_features=5),
    nn.Linear(in_features=5, out_features=5),
    nn.Linear(in_features=5, out_features=1)
)

#Loss and Optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_1.parameters(),
                            lr=0.01)

torch.manual_seed(42)
epoch = 1000

#Learning flux
for epoch in range(epoch):
    #Training the model
    model_1.train()

    #Forward pass
    y_predictions = model_1(X_train)

    #Loss
    loss = loss_fn(y_predictions, y_train)
    acc = accuracy_fn(y_train, y_predictions)
    
    #Zero grad
    optimizer.zero_grad()

    #Backwards
    loss.backward()

    #Optimizer step
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}")

#Ploting the model_1 predictions
model_1.eval()
with torch.inference_mode():
    y_pred = model_1(X_test)
    plt.scatter(X_train, y_train, c='b', label="Train")
    plt.scatter(X_test, y_test, c='g', label="Test")
    plt.scatter(X_test, y_pred, c='r', label="Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title('Linear model with neural network')
    plt.legend()

#%%
############## 3. USING A NON LINEAR FUNCTION TO MODELING THE SAME CIRCLE DATA ##############
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=1000,
                    noise=0.03,
                    random_state=42)
#Visualizing the data
#plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

'''
class CircularModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.layer_1(x))))

model_2 = CircularModelV2()

''' 

#Model using a non linear function, ReLU()
model_2 = nn.Sequential(
    nn.Linear(in_features=2, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_2.parameters(),
                            lr=0.1)

torch.manual_seed(42)
epochs = 10000

for epoch in range(epochs):
    #Training
    model_2.train()

    #Forward pass
    y_logits = model_2(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))

    #Loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_preds)

    #Zero grad
    optimizer.zero_grad()

    #Backwards
    loss.backward()

    #Step
    optimizer.step()
        
    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds) 

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}")

# Ploting the predictions
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Training")
plot_decision_boundary(model_2, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_2, X_test, y_test)

#%%
############## 4. MODELING A MULTICLASSIFICATION PROBLEM
from sklearn.datasets import make_blobs

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=2,
                            centers=4,
                            cluster_std=1.5,
                            random_state=42)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=42)

plt.figure(figsize=(10,7))
plt.scatter(X_blob[:,0], X_blob[:,1], c=y_blob, cmap=plt.cm.RdYlBu)

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model_3 = BlobModel(input_features=2,
                    output_features=4)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(), 
                            lr=0.1)

torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model_3.train()
    y_logits = model_3(X_blob_train)
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_blob_test, test_preds)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('Train')
plot_decision_boundary(model_3, X_blob_train, y_blob_train)
plt.subplot(1,2,2)
plt.title('Test')
plot_decision_boundary(model_3, X_blob_test, y_blob_test)