import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from rich.live import Live
from rich.table import Table
from utils import set_seed
import random
import torchvision.transforms as T
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from PIL import Image
import torchvision
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE


# live tables to view the training progress

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Training():

    """
    Training class for the model
    """
    def __init__(self):
        ...

    def train(self, model, trainloader, optimizer, criterion):

        model.train()

        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0

        for i, data in enumerate(trainloader):
            counter += 1

            # data files
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(image)

            # Calculate Loss
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()

            # Calculate Accuracy
            preds = outputs.argmax(axis=1)
            train_running_correct += (preds.squeeze() == labels).sum().item()

            # Backprop and optimize
            loss.backward()
            optimizer.step()

        # Epoch Loss and Accuracy
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(trainloader.sampler))

        return epoch_loss, epoch_acc

    def validate(self, model, testloader, criterion):

        model.eval()
        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0

        with torch.no_grad():
            for i, data in enumerate(testloader):
                counter += 1

                # Load data
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                # Forward Pass
                outputs = model(image)

                # Calculate loss
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()

                # Calculate accuracy
                preds = outputs.argmax(axis=1)
                valid_running_correct += (preds.squeeze()
                                          == labels).sum().item()

        # Epoch Loss and Accuracy
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(testloader.sampler))
        return epoch_loss, epoch_acc

    def fit(self, model, train_loader, valid_loader, epochs=4, lr=1e-4, save=False, show=False):
        table = Table()
        table.add_column("Epoch")
        table.add_column("Train Loss")
        table.add_column("Valid Loss")
        table.add_column("Train Accuracy")
        table.add_column("Valid Accuracy")

        set_seed(10)
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Track accuracy and loss
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        with Live(table, refresh_per_second=4) as live:
            # Loop through epochs
            for epoch in range(epochs):
                # Train model
                train_epoch_loss, train_epoch_acc = self.train(model, train_loader,
                                                            optimizer, criterion)

                # Validate model
                valid_epoch_loss, valid_epoch_acc = self.validate(model, valid_loader,
                                                                criterion)

                # Track training and validation
                train_loss.append(train_epoch_loss)
                valid_loss.append(valid_epoch_loss)
                train_acc.append(train_epoch_acc)
                valid_acc.append(valid_epoch_acc)

                table.add_row(str(epoch), str(round(train_epoch_loss,3)), str(round(valid_epoch_loss,3)), str(round(train_epoch_acc,3)), str(round(valid_epoch_acc,3)))
        

        
        # Save trained model
        if save:
            save_pth = './data/save_arnie.pth'
            torch.save(model.state_dict(),save_pth)

        # Visualize training progress
        if show:
            self.show_plots(train_acc, valid_acc, train_loss, valid_loss)

        print('Model Rendered')

    def show_plots(self, train_acc, valid_acc, train_loss, valid_loss):

        # Accuacy
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_acc, color='green', linestyle='-',
            label='train accuracy'
        )
        plt.plot(
            valid_acc, color='blue', linestyle='-',
            label='validataion accuracy'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-',
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-',
            label='validataion loss'
        )
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()


# class map for our dataset
class_map = {
     0: "enemy",
     1: "friend"
}

# mean and standard deviation for the Arnie dataset.

sq_mean = np.array([0.1859, 0.1772, 0.1768], dtype=np.float32) 
sq_std = np.array([0.1888, 0.1849, 0.1847], dtype=np.float32)


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Args:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    loss = model(X)[range(len(y)), y].sum()
    loss.backward()

    # As in paper, saliency is just the score grad with respect to input (abs+max)
    saliency, _ = X.grad.abs().max(axis=1)

    return saliency

def preprocess(img, size=128):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=sq_mean.tolist(),
                    std=sq_std.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def show_saliency_maps(X, y, model):
    # Convert X and y from numpy arrays to Torch Tensors
    X = np.array(X)
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    viz_lbl = [int(i) for i in y]
    y_tensor = torch.LongTensor(viz_lbl)
    viz_model = model
    for param in viz_model.parameters():
        param.requires_grad = False

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, viz_model.to('cpu'))

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_map[viz_lbl[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


# Define a function to extract intermediate layer activations
def get_intermediate_activations(model, dataloader, layer_name):
    intermediate_activations = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            activations = inputs
            for name, layer in model.named_children():
                if name == 'fc':
                    activations = activations.view(activations.size(0), -1)            
                else:
                    activations = layer(activations)
                if name == layer_name:
                    intermediate_activations.append(activations.view(activations.size(0), -1))
    return torch.cat(intermediate_activations)


def PCA_visualize(layer_name, model, test_loader): 
    # Get intermediate activations
    intermediate_activations = get_intermediate_activations(model, test_loader, layer_name)

    # Perform PCA for dimensionality reduction
    # pca = PCA(n_components=2)
    # reduced_activations = pca.fit_transform(intermediate_activations.view(intermediate_activations.size(0), -1))
    pca = PCA(n_components=3)
    reduced_activations = pca.fit_transform(intermediate_activations)


    # Create a scatter plot of the reduced activations
    plt.scatter(reduced_activations[:, 0], reduced_activations[:, 1], c='b', marker='o')
    plt.title(f'PCA Visualization of Layer {layer_name} Activations')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


def PCA_visualization_sns(layer_name, model, test_loader):
    intermediate_activations = get_intermediate_activations(model, test_loader, layer_name)
    labels = []
    for batch in test_loader:
        _, batch_labels = batch  # Assuming the labels are in the second element of each batch
        labels.extend(batch_labels.tolist()) 

    pca = PCA(n_components=2)
    reduced_activations = pca.fit_transform(intermediate_activations)
    print(reduced_activations.shape)

    principalDf =pd.DataFrame(data = reduced_activations, columns = ['principalcomponent1',  'principalcomponent2'])
    label = pd.DataFrame(list(labels))
    principalDf = pd.concat([principalDf,label],axis = 1,join='inner', ignore_index=True)
    principalDf = principalDf.loc[:,~principalDf.columns.duplicated()]
    principalDf.columns = ["principalcomponent1", "principalcomponent2", "label"] 

    sns.color_palette("YlOrBr", as_cmap=True)
    sns.lmplot( x="principalcomponent1", y="principalcomponent2", data=principalDf, fit_reg=False,
            hue='label', legend=True)

    plt.figure(figsize=(13,10))


def TSNE_visualize(layer_name, model, test_loader):
    intermediate_activations = get_intermediate_activations(model, test_loader, layer_name)
    labels = []
    for batch in test_loader:
        _, batch_labels = batch  # Assuming the labels are in the second element of each batch
        labels.extend(batch_labels.tolist()) 
    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30)
    reduced_activations = tsne.fit_transform(intermediate_activations)

    # Get labels from the dataloader
    labels = np.array([])
    for _, batch_labels in test_loader:
        labels = np.concatenate((labels, batch_labels.numpy()))

    # Get unique class labels
    unique_labels = np.unique(labels)

    # Create a DataFrame for your data
    data = pd.DataFrame(data=reduced_activations, columns=['Component 1', 'Component 2'])
    data['Label'] = labels


    # Create a scatter plot using Seaborn with the correct palette
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Component 1', y='Component 2', hue='Label', data=data)
    plt.title(f't-SNE Visualization of Layer {layer_name} Activations')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Class', loc='upper right')
    plt.show()

