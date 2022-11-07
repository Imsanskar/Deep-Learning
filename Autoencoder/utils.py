import torch
import torch.nn as nn
import torchvision

class AutoEncoder(nn.Module):
    def __init__(self, code_size:int):
        self.code_size = code_size
        super(AutoEncoder, self).__init__()
        self.encoder =  nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, code_size)
        )
        
        self.decoder =  nn.Sequential(
            nn.Linear(code_size, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_space = self.encoder(x)
        op = self.decoder(latent_space)
        return torch.reshape(op, (-1, 1, 28, 28))

    def get_latent_space(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return torch.reshape(self.decoder(x), (-1, 1, 28, 28))


"""
    Returns train and test dataloader
"""
def load_mnist_dataset(train_batch_size, test_batch_size):
    train_dataset = torchvision.datasets.MNIST(
        "./data/mnist/", 
        train = True, 
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    )
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = train_batch_size)
    test_dataset = torchvision.datasets.MNIST(
        "./data/mnist/", 
        train = True, 
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    )
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size)

    return train_data_loader, test_data_loader

def train(model: nn.Module, data: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, epochs = 10, device = 'cpu', writer = None):
    loss_fn = nn.MSELoss()
    model.to(device)
    for epoch in range(epochs):
        i = 0
        error = 0
        for inp, label in data:
            inp = inp.to(device)
            optimizer.zero_grad()
            output = model(inp)
            loss = loss_fn(inp, output.to(device))
            loss.backward()
            optimizer.step()
            error += torch.linalg.norm(loss).item()
            i += 1
        if writer != None:
            writer.add_scalar("Loss/train", error / i, epoch + 1)
        print(f"Error for epoch {epoch + 1}: {error / i}")

    return model