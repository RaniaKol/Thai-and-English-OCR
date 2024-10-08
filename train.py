import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from model import OCRModel 


class ToTensor:
    def __call__(self, image):
        return torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return cv2.resize(image, self.size)


class ImDataset(Dataset):
    def __init__(self, data_tuples, transform=None):
        self.label_mapping = self.create_label_mapping(data_tuples)
        self.data = self.load_data(data_tuples)
        self.transform = transform

    def create_label_mapping(self, data_tuples):
        label_mapping = {}
        for entry in data_tuples:
            if len(entry) != 5:
                print(f"Warning: Entry format is incorrect: {entry}")
                continue
            
            _, _, _, class_name, _ = entry  
            class_name = class_name.strip().strip("'") 
            
            if class_name not in label_mapping:
                label_mapping[class_name] = len(label_mapping)

        return label_mapping

    def load_data(self, data_tuples):
        data = []
        for entry in data_tuples:
            if len(entry) != 5:
                print(f"Warning: Entry format is incorrect: {entry}")
                continue
        
            language, dpi, style, class_name, image_path = entry

            class_name = class_name.strip().strip("'") 
            image_path = image_path.strip().strip("'").strip(")").rstrip("'")
            if class_name not in self.label_mapping:
                print(f"Warning: Class name '{class_name}' not in label mapping. Entry: {entry}")
                continue
            
            mapped_label = self.label_mapping[class_name]
            full_image_path = Path(image_path) 
            data.append((full_image_path, mapped_label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        if not isinstance(image_path, Path):
            image_path = Path(image_path)

        if not image_path.exists():
            print(f"Error: Path does not exist - {image_path}")
            raise FileNotFoundError(f"Image file not found at {image_path}")


        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return image, label_tensor


def load_data_from_file(file_path):
    data_tuples = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split(',')
            if len(parts) == 5:  
                entry = (parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip())
                data_tuples.append(entry)
            else:
                print(f"Warning: Invalid tuple format: {line}")
    return data_tuples


# Train the Model
def train_model(train_data_file, val_data_file, batch_size, num_epochs, save_dir, log_file):
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training run")

    # Load data
    train_data_tuples = load_data_from_file(train_data_file)
    val_data_tuples = load_data_from_file(val_data_file)

    transform = transforms.Compose([
        Resize((128, 128)), 
        ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImDataset(train_data_tuples, transform=transform)
    val_dataset = ImDataset(val_data_tuples, transform=transform)

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Stop training.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    num_classes = len(train_dataset.label_mapping)
    logger.info(f"Detected {num_classes} classes in the dataset.")

    model = OCRModel((128, 128), num_classes)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}], Loss: {loss.item():.4f}")
            running_loss += loss.item()

    # Save the model
    torch.save(model.state_dict(), Path(save_dir) / "model.pth")
    logger.info(f"Model saved to {Path(save_dir) / 'model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--tr_set", type=Path, required=True, help="Path to the training set.")
    parser.add_argument("--val_set", type=Path, required=True, help="Path to the validation set.")
    parser.add_argument("--batch", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs.")
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to save the model.")
    parser.add_argument("--log_file", type=Path, required=True, help="Path to the log file.")

    args = parser.parse_args()

    # Train the model
    train_model(args.tr_set, args.val_set, args.batch, args.epochs, args.save_dir, args.log_file)
