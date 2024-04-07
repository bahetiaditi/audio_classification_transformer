import os
import torch
from torch import nn, optim
import pytorch_lightning as pl
from dataset import CustomDataModule
from model import SoundClassifier, AudioClassifierWithTransformer
from utils import save_model, load_model, test_model, plot_confusion_matrix
from .experiments import arch1_exp1, arch1_exp2, arch2_exp1, arch2_exp2, arch2_exp3, arch2_exp4


def train_one_epoch(model, train_loader, criterion, optimizer, device, max_norm=2.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

#Used for CNN architecture
def trains_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(train_loader), 100 * total_correct / total_samples

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(val_loader), 100 * total_correct / total_samples

def test_model(model, test_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct_predictions / total_predictions
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
    roc_auc = roc_auc_score(all_labels_binarized, all_probabilities, multi_class='ovr', average='macro')

    plot_confusion_matrix(conf_matrix, classes=range(num_classes), title='Confusion Matrix')
    wandb.log({"Confusion Matrix": wandb.Image(plt)})

    fpr, tpr, roc_auc_dict = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], np.array(all_probabilities)[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'gray'])
    plt.figure()
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (area = {roc_auc_dict[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multi-Class)')
    plt.legend(loc="lower right")
    wandb.log({"ROC Curve": wandb.Image(plt)})
    plt.close()

    return avg_loss, accuracy, f1, roc_auc

def main():
    arch1_exp1()
    arch1_exp2()
    arch2_exp1()
    arch2_exp2()
    arch2_exp3()
    arch2_exp4()

if __name__ == "__main__":
    main()