import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Чтение конфигурации из файла
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Определение простой нейронной сети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(config['model']['input_size'], config['model']['hidden_size'])
        self.fc2 = nn.Linear(config['model']['hidden_size'], config['model']['output_size'])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_and_preprocess_data():
    data = load_iris()
    X = data.data
    y = data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state'])
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, criterion, optimizer, epochs=1000):
    for epoch in range(epochs):
        model.train()
        inputs, labels = Variable(X_train), Variable(y_train)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = accuracy_score(y_test, predicted)
        f1 = f1_score(y_test, predicted, average='weighted')
        precision = precision_score(y_test, predicted, average='weighted')
        recall = recall_score(y_test, predicted, average='weighted')
    return accuracy, f1, precision, recall

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    logger.info(f"Model saved to {filename}")

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    logger.info(f"Model loaded from {filename}")

def fgsm_attack(model, loss, data, target, epsilon):
    data, target = Variable(data, requires_grad=True), Variable(target)
    output = model(data)
    model.zero_grad()
    cost = loss(output, target)
    cost.backward()
    attack_data = data + epsilon * data.grad.sign()
    return attack_data

def adversarial_training(model, X_train, y_train, criterion, optimizer, epsilon, epochs=1000):
    adversarial_X_train = fgsm_attack(model, criterion, X_train, y_train, epsilon)
    augmented_X_train = torch.cat([X_train, adversarial_X_train])
    augmented_y_train = torch.cat([y_train, y_train])
    train_model(model, augmented_X_train, augmented_y_train, criterion, optimizer, epochs)

def visualize_results(accuracies, labels):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Condition')
    plt.show()

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    train_model(model, X_train, y_train, criterion, optimizer, epochs=config['training']['epochs'])
    initial_accuracy, initial_f1, initial_precision, initial_recall = evaluate_model(model, X_test, y_test)
    logger.info(f"Initial model accuracy: {initial_accuracy}")
    logger.info(f"Initial model F1-score: {initial_f1}")
    logger.info(f"Initial model Precision: {initial_precision}")
    logger.info(f"Initial model Recall: {initial_recall}")
    save_model(model, "initial_model.pth")

    X_adversarial = fgsm_attack(model, criterion, X_test, y_test, config['attack']['epsilon'])
    adversarial_accuracy, adversarial_f1, adversarial_precision, adversarial_recall = evaluate_model(model, X_adversarial, y_test)
    logger.info(f"Accuracy after FGSM attack: {adversarial_accuracy}")
    logger.info(f"F1-score after FGSM attack: {adversarial_f1}")
    logger.info(f"Precision after FGSM attack: {adversarial_precision}")
    logger.info(f"Recall after FGSM attack: {adversarial_recall}")

    poisoned_X_train = X_train.clone()
    poisoned_y_train = y_train.clone()
    poisoned_X_train[:config['attack']['num_poisoned_samples']] += torch.randn(config['attack']['num_poisoned_samples'], X_train.shape[1]) * 5
    poisoned_model = Net()
    optimizer = optim.Adam(poisoned_model.parameters(), lr=config['training']['learning_rate'])
    train_model(poisoned_model, poisoned_X_train, poisoned_y_train, criterion, optimizer, epochs=config['training']['epochs'])
    poisoned_accuracy, poisoned_f1, poisoned_precision, poisoned_recall = evaluate_model(poisoned_model, X_test, y_test)
    logger.info(f"Accuracy after data poisoning: {poisoned_accuracy}")
    logger.info(f"F1-score after data poisoning: {poisoned_f1}")
    logger.info(f"Precision after data poisoning: {poisoned_precision}")
    logger.info(f"Recall after data poisoning: {poisoned_recall}")

    extracted_data = []
    with torch.no_grad():
        for x in X_test:
            extracted_data.append(poisoned_model(x.unsqueeze(0)).numpy())
    logger.info(f"Extracted data (first 5 predictions): {extracted_data[:5]}")

    robust_model = Net()
    optimizer = optim.Adam(robust_model.parameters(), lr=config['training']['learning_rate'])
    adversarial_training(robust_model, X_train, y_train, criterion, optimizer, config['attack']['epsilon'], epochs=config['training']['epochs'])
    robust_accuracy, robust_f1, robust_precision, robust_recall = evaluate_model(robust_model, X_test, y_test)
    logger.info(f"Accuracy of robust model: {robust_accuracy}")
    logger.info(f"F1-score of robust model: {robust_f1}")
    logger.info(f"Precision of robust model: {robust_precision}")
    logger.info(f"Recall of robust model: {robust_recall}")
    robust_adversarial_accuracy, robust_adversarial_f1, robust_adversarial_precision, robust_adversarial_recall = evaluate_model(robust_model, X_adversarial, y_test)
    logger.info(f"Robust model accuracy after adversarial attack: {robust_adversarial_accuracy}")
    logger.info(f"Robust model F1-score after adversarial attack: {robust_adversarial_f1}")
    logger.info(f"Robust model Precision after adversarial attack: {robust_adversarial_precision}")
    logger.info(f"Robust model Recall after adversarial attack: {robust_adversarial_recall}")

    accuracies = [initial_accuracy, adversarial_accuracy, poisoned_accuracy, robust_adversarial_accuracy]
    labels = ['Initial', 'Adversarial Attack', 'Data Poisoning', 'Robust']
    visualize_results(accuracies, labels)

if __name__ == "__main__":
    main()
