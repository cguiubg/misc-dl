import torch
import matplotlib.pyplot as plt

from mnist_autoencoder import test_loader
from mnist_autoencoder import Autoencoder

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
untrained_network = Autoencoder().to(device)
trained_network = Autoencoder().to(device)
trained_network.load_state_dict(torch.load("./weights/weights-31.pt"))

batch_idx, (data, label) = next(enumerate(test_loader))
with torch.no_grad():
    img_batch = data.to(device)
    untrained_decoded = untrained_network(img_batch)
    trained_decoded = trained_network(img_batch)

untrained_img = untrained_decoded[0].reshape((28, 28)).cpu()
trained_img = trained_decoded[0].reshape((28, 28)).cpu()
training_img = data[0].reshape((28,28))
plt.figure(figsize = (23, 8))
##
plt.subplot(1, 3, 1)
plt.imshow(untrained_img, cmap='gray')
plt.title('Image (Untrained network), Label ' + str(label[0].item()), fontsize = 15)
plt.axis("off")
##
plt.subplot(1, 3, 2)
plt.imshow(trained_img, cmap='gray')
plt.title('Image (Trained network), Label ' + str(label[0].item()), fontsize = 15)
plt.axis("off")
###
plt.subplot(1, 3, 3)
plt.imshow(training_img, cmap='gray')
plt.title('Training Image, Label ' + str(label[0].item()), fontsize = 15)
plt.axis("off")

plt.show()
