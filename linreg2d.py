import torch
import numpy as np
import matplotlib.pyplot as plt
import csv




class LinearRegressionModel:

    def __init__(self, data):
        self.features = torch.tensor(data[:, 0], dtype=torch.float32).view(-1, 1)
        self.targets = torch.tensor(data[:, 1], dtype=torch.float32).view(-1, 1)
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

data = []

with open('data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  
    for row in csvreader:
        num1, num2 = map(float, row)
        data.append([num1, num2])

data = np.array(data)

model = LinearRegressionModel(data)

learning_rate = 0.0001
num_epochs = 1000


optimizer = torch.optim.SGD([model.W, model.b], lr=learning_rate)

for epoch in range(num_epochs):
    model.W.grad = None
    model.b.grad = None

    
    loss = model.loss(model.features, model.targets)

    
    loss.backward()

   
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

       
        plt.figure()
        plt.scatter(model.features.numpy(), model.targets.numpy(), label='Data')
        plt.plot(model.features.numpy(), model.f(model.features).detach().numpy(), color='red', label='Regression Line')
        plt.xlabel('Length')
        plt.ylabel('Weight')
        plt.legend()
        plt.show()
