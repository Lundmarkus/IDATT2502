import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

class LinearRegressionModel:

    def __init__(self, data):
        self.features = torch.tensor(data[:, :2], dtype=torch.float32)
        self.targets = torch.tensor(data[:, 2], dtype=torch.float32).view(-1, 1)
        self.W = torch.tensor([[0.0], [0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W[:-1] + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


data = []

with open('day_length_weight.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    
    next(csv_reader, None)
    
    for row in csv_reader:
       
        if len(row) >= 3:
            x1 = float(row[0])
            x2 = float(row[1])
            y = float(row[2])
            data.append([x1, x2, y])
        else:
            print(f"Skipping row with insufficient values: {row}")

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


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='Data', c='blue', marker='o')


x = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
y = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)
X, Y = np.meshgrid(x, y)
xy_grid = np.column_stack((X.flatten(), Y.flatten()))


xy_grid = np.column_stack((xy_grid, np.ones(xy_grid.shape[0])))


Z = model.f(torch.tensor(xy_grid, dtype=torch.float32)).reshape(X.shape)


ax.plot_surface(X, Y, Z.detach().numpy(), cmap='viridis', alpha=0.7)


ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target (Feature 3)')


plt.show()
