import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision

import matplotlib.pyplot as plt


batch_size_train = 32
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./files/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=True
)


batch_idx, (example_data, example_target) = next(enumerate(train_loader))
print(example_data.shape)

nef = ndf = 32
n_channels = 1

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # in: [1, 28, 28]
        self.conv1 = nn.Conv2d(n_channels, 10, kernel_size=5)
        # : [10, 14, 14]
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # : [20, 7, 7]
        self.dense1 = nn.Linear(20 * 4 * 4, 128)
        self.dense2 = nn.Linear(128, 32)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
        x = F.leaky_relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4 * 4 * 20)
        x = F.leaky_relu(self.dense1(x))
        x = self.dense2(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(32, 128)
        self.dense2 = nn.Linear(128, 20 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(10, n_channels, kernel_size=5, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = x.view(-1, 20, 4, 4)
        x = F.leaky_relu(self.conv1(x, output_size=(-1, 10, 12, 12)))
        x = F.tanh(self.conv2(x, output_size=(-1, 1, 28, 28)))
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
###############################################################################

if __name__ == "__main__":
    lr = 0.001
    weight_decay = 0
    num_epochs = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = Autoencoder().to(device)
    opt = torch.optim.Adam(network.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn = nn.MSELoss().to(device)

    print("________ Network ________")
    print(network)
    print("_________________________")

    losses = []
    save_on = 8

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            img_batch = data.to(device)
            decoded = network(img_batch)
            loss = loss_fn(decoded, img_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch: {epoch}, Loss {epoch_loss:.4f}")

        if (epoch + 1) % save_on == 0:
            torch.save(network.state_dict(), f"./weights/weights-{epoch}.pt")

    print("\n____________________________________________________\n")

    fig = plt.figure(figsize = (12,5))

    plt.plot(losses, '-r', label='Training loss')
    plt.xlabel('Epochs', fontsize= 15)
    plt.ylabel('Loss', fontsize= 15)
    plt.title('Convolutional AutoEncoder Training Loss Vs Epochs', fontsize= 15)
    plt.show()

    print("\n____________________________________________________\n")