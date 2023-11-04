from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import cv2
import random
import glob
import json
from tqdm import tqdm
import pathlib
from typing import List
import pandas as pd
import requests
import cv2

# Device configuration
def get_device():

    """
    Returns the 'cuda' if GPU is available on the system
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def set_seed(seed: int):
    """
    Seed function for reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class Tenyks_data(Dataset):

    def __init__(self, data_dir,class_mapping, transform=None):
        
        self.data_dir = data_dir
        self.transform = transform
        file_list = glob.glob(self.data_dir + "/*")
        # print(len(file_list))
        self.paths = list(pathlib.Path(data_dir).glob("*/*.jpg"))

        self.data = []
        with open(class_mapping, "r") as file: dictionary = json.load(file)
        self.class_map = dictionary
        # print(self.class_map)

        for class_pth in file_list:
            class_name = class_pth.split("/")[-1]
            for img_pth in glob.glob(class_pth + "/*.jpg"):
                self.data.append([img_pth, int(class_name)])
        # print(len(self.data))
        
    def __len__(self):
        return len(self.data)
    
    def load_image(self, idx) -> Image.Image:
       image_pth = self.path[idx]
       return Image.open(image_pth)

    def __getitem__(self, index):
        image_pth, label = self.data[index]
        # label = self.class_map[label] # we get error while sending the data to GPU
        image = Image.open(image_pth)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

def walk_through_dir(dir_path):
  """
  Walks through dir_path to get the number of files in directory.

  Args:
    dir_path (str or pathlib.Path): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    
    if len(dirnames) > 0:
      print(f"There are {len(dirnames)} directories.")

    elif len(filenames) >0:
        type_dataset = dirpath.split('/')[-2]
        category = dirpath.split('/')[-1]
        print(f"There are {len(filenames)} images in {type_dataset}, class'{category}'.")



def visualize_random_images(train_dir, num_images_per_class=10):
    
    """
    Args:
        train_dir : takes the directory of the dataset for visualization random images.
        num_imags_per_class : pass in the number of images to visualize.
    Returns:
        A plot of num_images_per_class images from each class in train_dir.
    """

    class_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

    for class_folder in class_folders:
        class_path = os.path.join(train_dir, class_folder)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        num_images_to_visualize = min(num_images_per_class, len(image_files))

        if num_images_to_visualize == 0:
            continue

        print(f"Class: {class_folder}")
        fig, axes = plt.subplots(1, num_images_to_visualize, figsize=(16, 8))

        for i in range(num_images_to_visualize):
            random_image_file = random.choice(image_files)
            image_path = os.path.join(class_path, random_image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            axes[i].imshow(image)
            axes[i].set_title(f"{i + 1}")
            axes[i].axis("off")

        plt.show()

def view_random_images(path_for_images):
    # Directory containing the images


    # List all image files in the folder
    image_files = [f for f in os.listdir(path_for_images)]

    # Randomly select 5 images
    selected_images = random.sample(image_files, 5)  # Change 10 to 5

    # Set up a Matplotlib figure to display the images
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))  # Change 2 to 1 and 5 images

    # Loop through and display the selected images
    for i, image_file in enumerate(selected_images):
        image_path = os.path.join(path_for_images, image_file)
        image = cv2.imread(image_path)

        # Convert BGR image to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i].imshow(image_rgb)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



def get_random_images(train_dir, num_images_per_class=5):

    class_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

    numpy_images_arr = []
    numpy_lbl_arr = []
    for class_folder in class_folders:
        class_path = os.path.join(train_dir, class_folder)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        num_images_to_visualize = min(num_images_per_class, len(image_files))

        if num_images_to_visualize == 0:
            continue

        for i in range(num_images_to_visualize):
            random_image_file = random.choice(image_files)
            image_path = os.path.join(class_path, random_image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            numpy_images_arr.append(image)
            numpy_lbl_arr.append(class_folder)
        
    return numpy_images_arr, numpy_lbl_arr


def display_random_images(dataset: Dataset, classes: List[str] = None, n: int = 10, display_shape: bool = True, seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print("For display purposes, n shouldn't be larger than 10. Setting n to 10 and removing shape display.")

    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16, 8))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample]

        targ_image_adjust = targ_image.permute(1, 2, 0)

        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        print(targ_label)
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
            plt.title(title)

    plt.show()


def check_image_extensions(pth):

    """
    small quick checker for the extension of the images!
    """

    for dirpath, dirnames, filenames in os.walk(pth):
        img_ext = []
        for img in filenames:
            img_ext.append(img.split('.')[-1])
        print(set(img_ext))


def show_transformed_img(original_image_path):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((185, 185)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor()
    ])

    # Load the original image
    original_image = Image.open(original_image_path)

    # Apply the transformation
    transformed_image = transform(original_image)

    # Create subplots to display both images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image on the left subplot
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display the transformed image on the right subplot
    # Note: You may need to convert the transformed image to a NumPy array
    transformed_image_np = transformed_image.permute(1, 2, 0).numpy()
    axes[1].imshow(transformed_image_np)
    axes[1].set_title("Transformed Image")
    axes[1].axis('off')

    # Show the subplots
    plt.show()



def compute_mean_std(dataset_obj):
    N_CHANNELS = 3
    full_loader = DataLoader(dataset_obj, shuffle=False, num_workers=4)

    # Initialize mean and std tensors with a size of N_CHANNELS
    mean = torch.zeros(N_CHANNELS)
    std = torch.zeros(N_CHANNELS)

    print('==> Computing mean and std..')
    for inputs, _ in tqdm(full_loader):
        for i in range(N_CHANNELS):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    mean.div_(len(dataset_obj))
    std.div_(len(dataset_obj))

    return mean, std


def download_images(csv_file, output_folder):
    # Path to your CSV file containing Google Images URLs
    csv_file = csv_file

    # Output folder to save the downloaded images
    output_folder = output_folder

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file into a pandas DataFrame without specifying headers
    df = pd.read_csv(csv_file, header=None)

    # Loop through the URLs in the DataFrame and download the images
    for index, row in df.iterrows():
        image_url = row.iloc[0]  # Since there are no headers, use iloc[0] to access the first column
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                # Extract the file name from the URL and save the image
                image_name = os.path.join(output_folder, os.path.basename(image_url))
                with open(image_name, 'wb') as image_file:
                    image_file.write(response.content)
                print(f"Downloaded: {image_name}")
            else:
                print(f"Failed to download: {image_url}")
        except Exception as e:
            print(f"Error while downloading {image_url}: {str(e)}")

    print("Download process completed.")



def create_extension(folder_path):

    image_folder = folder_path

    # List all files in the folder
    files = os.listdir(image_folder)

    # Loop through the files and add the .jpg extension to those without extensions
    for file in files:
        if not file.endswith(".jpg"):
            new_name = file + ".jpg"
            os.rename(os.path.join(image_folder, file), os.path.join(image_folder, new_name))
            print(f"Renamed: {file} to {new_name}")

    print("Extension addition completed.")


def create_torch_data(folder_path):
    image_data = []
    for filename in os.listdir(folder_path):
         # Adjust extensions as needed
        # Open the image using PIL
        img = Image.open(os.path.join(folder_path, filename))

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Append the image array to the list
        image_data.append(img_array)
    
    image_tensor = np.stack(image_data)
    image_tensor = torch.from_numpy(image_tensor)
    
    return image_tensor


