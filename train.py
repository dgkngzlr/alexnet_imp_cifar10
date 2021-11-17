import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from alexnet import AlexNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import numpy as np

def get_test_acc(model):
  
  sm = nn.Softmax(1)
  scores = [] 
  model.eval()

  with torch.no_grad():

    for step, (inputs, labels) in enumerate(test_loader):

      inputs, labels = inputs.to(device), labels.to(device)

      y_hat = model.forward(inputs)
      y_hat = sm(y_hat)
      y_pred = torch.argmax(y_hat, axis=1)
    
      y_pred = y_pred.to("cpu").clone().numpy().reshape(-1,1)
      y_test = labels.to("cpu").clone().numpy().astype(np.int32).reshape(-1,1)

      cm = confusion_matrix(y_test, y_pred)
      score = ((np.eye(10) * cm).sum() / cm.sum())*100
      scores.append(round(score,2))
  
  model.train()
  return np.mean(scores)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = T.Compose([T.ToTensor(),
                       T.Resize(256),
                       T.CenterCrop(227),
                       T.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124],std=[0.24703233,0.24348505,0.26158768])
                       ])

train_dataset = torchvision.datasets.CIFAR10("./CIFAR10/train", train=True, \
                                             transform=transform, download=True)

test_dataset = torchvision.datasets.CIFAR10("./CIFAR10/test", train=False, \
                                             transform=transform, download=True)

classes = list(train_dataset.class_to_idx.keys())


fig,axs = plt.subplots(nrows=5, ncols=5, figsize=(12,9))

k = 0
for i in range(5):
    for j in range(5):
        data, label = train_dataset[k]
        axs[i,j].set_title(classes[label].title())
        axs[i,j].imshow(data.permute(1,2,0))
        k += 1

plt.tight_layout()
plt.show()


model = AlexNet()
model.train()
model.to(device)


n_epoch = 100
lr = 0.01
weight_decay = 5e-4
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

criterion = nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9, weight_decay=weight_decay)

summary(model, (3,227,227))

epoch_losses = []
test_accs = []
best_acc = 0
prev_loss = 1e4
for ep in range(n_epoch):

    step_losses = []
    for step, (inputs, labels) in enumerate(train_loader):

        inputs, labels = inputs.to(device), labels.to(device)
    
        y_hat = model.forward(inputs)
        
        loss = criterion(y_hat, labels)
        step_losses.append(loss.item())
        if (step+1) % 100 == 0:
          print(f"Epoch {ep+1} Step => {step+1} Loss => {step_losses[step]}")
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    epoch_losses.append(np.mean(step_losses))
    step_losses.clear()
    test_acc = get_test_acc(model)
    test_accs.append(test_acc)

    if test_acc >= best_acc and epoch_losses[ep] < prev_loss:
              best_acc = test_acc
              prev_loss = epoch_losses[ep]
              torch.save(model.state_dict(), "alexnet_cifar10_best.pth")
              print("Best model saved !")
    if (ep+1) % 1 == 0:
              
              print(f"Epoch {ep+1} Step => {step} Loss => {epoch_losses[ep]} Test_Acc(%) => {test_acc} Best(%) => {best_acc}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(epoch_losses)
plt.show()

plt.xlabel("Epoch")
plt.ylabel("Test Acc")
plt.plot(test_accs)
plt.show()