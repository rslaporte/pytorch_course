#%%
'''
    Exercise 1: Make a binary classification using scikit_learn make_moon dataset, with n_samples=1000
    - Use test_size = 20% of the sample
    - Make the model reach over 96% accuracy.
    - Plot the trained model bondaries    
'''
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchmetrics import Accuracy
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

X, y = make_moons(n_samples=1000,
                  noise=0.08,
                  random_state=42)

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.RdYlBu)


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=42,
                                                    test_size=0.2)

class MoonModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=output_features)
        )
    
    def forward(self, x):
        return self.layers(x)

model_moon = MoonModel(input_features=2, 
                       output_features=1)

loss_fn = nn.BCEWithLogitsLoss()
acc_fn = Accuracy(task="binary")
optimizer = torch.optim.SGD(model_moon.parameters(),
                            lr=0.1)

#%%
#Veryfing if everything is ok
model_moon.eval()
with torch.inference_mode():
    y_logits = model_moon(X_train).squeeze()
    print(torch.sigmoid(y_logits))

#%%
torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    model_moon.train()

    y_logits = model_moon(X_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)

    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred_probs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_moon.eval()
    with torch.inference_mode():
        y_logits_test = model_moon(X_test).squeeze()
        y_pred_probs_test = torch.sigmoid(y_logits_test)

        test_loss = loss_fn(y_logits_test, y_test)
        test_acc = acc_fn(y_pred_probs_test, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}")

#%%
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_moon, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_moon, X_test, y_test)

#%%
'''
    Exercise 2: Recriate the hyperbolic tangent function using pure Pytorch
'''
def tanh(x):
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))

X = torch.arange(-10,10,0.2)
y = tanh(X)

X[:10], y[:10]
#%%
plt.title("Hyberbolic Tangent")
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, y, label=r'$y = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}$')
plt.legend()

#%%
'''
    Exercise 3: Using the spirals data creation function from CS231 (given), create a multi-class
    model with accuracy higher than 95%.
    - Use 1000 epochs
    - Use Adam optimizer
'''

def spiral_dataset():
    np.random.seed(42)
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return [X, y]

X, y = spiral_dataset()

# Visualizing
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

X_test, X_train, y_test, y_train = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

class SpiralModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=hidden_layers),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layers, out_features=output_features)
        )

    def forward(self, x):
        return self.layers(x)
    
model_spiral = SpiralModel(2, 3)
loss_fn = nn.CrossEntropyLoss()
acc_fn = Accuracy(task='multiclass', num_classes=3)
optimizer = torch.optim.Adam(model_spiral.parameters(), 
                             lr=0.01)

#%%
#Primary evaluation
model_spiral.eval()
with torch.inference_mode():
    y_logits = model_spiral(X_test)
    y_pred_probs = torch.argmax(y_logits, dim=1)
    print(y_pred_probs)    

#%%
torch.manual_seed(42)
epochs = 1000

for epoch in range(epochs):
    model_spiral.train()

    y_logits = model_spiral(X_train)
    y_pred_probs = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred_probs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_spiral.eval()
    with torch.inference_mode():
        y_logits_test = model_spiral(X_test)
        y_pred_probs_test = torch.softmax(y_logits_test, dim=1).argmax(dim=1)

        test_loss = loss_fn(y_logits_test, y_test)
        test_acc = acc_fn(y_pred_probs_test, y_test)

    if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f} | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}")

#%%
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_spiral, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_spiral, X_test, y_test)