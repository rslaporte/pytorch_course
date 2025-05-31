# %%
import torch
import numpy as np
from matplotlib import pyplot as plt

############################### SPLITING DATA
# %%
weight = 0.7
bias = 0.3

# %%
start = 0
end = 1
step = 0.02
X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

# %%
train_split = int(0.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# %%
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))

    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')

    if(predictions is not None):
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={"size":14})

############################### MODELING
# %%
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                  requires_grad=True,
                                  dtype=torch.float))
        
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias
# %%
torch.manual_seed(42)
model_0 = LinearRegressionModel()

# %%
with torch.inference_mode():
    y_preds = model_0(X_test)

# %%
list(model_0.parameters())

############################### TRAINING
# %%
#Loss Function
loss_fn = nn.L1Loss()

#Optmizer
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

#%%
test_loss_count = []
loss_count = []
epoch_count = []
epochs = 100

for epoch in range(epochs):
    #1. Training
    model_0.train()

    #2. Forward pass
    y_pred = model_0(X_train)

    #3. Loss
    loss = loss_fn(y_pred, y_train)

    #4 - Optimizer zero
    optimizer.zero_grad()

    #5 - Backpropagation
    loss.backward()

    #6 - Optimizer step
    optimizer.step()

    model_0.eval() #Turns off the gradient
    with torch.inference_mode(): 
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

        epoch_count.append(epoch)
        loss_count.append(loss)
        test_loss_count.append(test_loss)

    #if epoch % 10 == 0 or epoch == 99:
        #print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}")
        #print(model_0.state_dict())
# %%
plt.plot(epoch_count, np.array(torch.tensor(loss_count).numpy()), label="Train Loss")
plt.plot(epoch_count, np.array(torch.tensor(test_loss_count).numpy()), label="Test Loss")
plt.title("Traning and Test Loss curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# %%
with torch.inference_mode():
    y_preds_new = model_0(X_test)

plot_predictions(predictions=y_preds)
plot_predictions(predictions=y_preds_new)

#%%
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "pytorch_worflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), 
           f=MODEL_SAVE_PATH)

# %%
loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_0.state_dict()

#%%
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

loaded_model_preds