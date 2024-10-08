import torch
from torch.utils.data import DataLoader
import argparse
from train import ImDataset
from model import OCRModel
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import numpy as np
import torchvision.transforms as transforms

def load_model(model_path, res, label):
    model = OCRModel(res, label)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(test_data_file, model, batch_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: cv2.resize(img, (128, 128))),
        transforms.Lambda(lambda img: torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    test_data_tuples = load_data_from_file(test_data_file)
    test_dataset = ImDataset(test_data_tuples, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels

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

def main(test_data_file, model_path, batch_size):
    test_data_tuples = load_data_from_file(test_data_file)
    test_dataset = ImDataset(test_data_tuples) 

    img_res = (128, 128)
    n_labels = len(test_dataset.label_mapping)

    model = load_model(model_path, img_res, n_labels)
    predictions, labels = evaluate_model(test_data_file, model, batch_size)

    # Calculate overall accuracy
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, help="Path to the test data")
    parser.add_argument("--model_path", type=str, help="Path to the saved model")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()
    main(args.test_set, args.model_path, args.batch)
