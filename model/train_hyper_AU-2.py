import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from data_EH import get_dataloaders
from data_AU import get_dataloaders_au, get_class_weights
from model import ResNetModel_IFOF, EfficientNetModel_IFOF

# Function to define and optimize hyperparameters
def objective(trial):
    # Define hyperparameters to search over
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    
    # Configuration
    num_epochs = 100  # Adjust as needed
    image_dir = '/home/rdas/student_engagement/WACV'
    feature_dir = '/home/rdas/student_engagement/WACV'

    # Load Datasets
    train_loader_EH, val_loader_EH, _ = get_dataloaders(image_dir, feature_dir, batch_size)
    train_loader_AU, val_loader_AU, _ = get_dataloaders_au(image_dir, feature_dir, batch_size)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get class weights for IFOF model and pass them to the loss function
    train_labels = np.array([train_loader_AU.dataset.dataset.labels[i] for i in train_loader_AU.dataset.indices])
    class_weights = torch.tensor(get_class_weights(train_labels)).float().to(device)

    # Define models
    modelResNet_EH = ResNetModel_IFOF(image_input_dim=512, feature_input_dim=294).to(device)
    modelResNet_AU = ResNetModel_IFOF(image_input_dim=512, feature_input_dim=35).to(device)
    modelEfficientNet_EH = EfficientNetModel_IFOF(image_input_dim=1280, feature_input_dim=294).to(device)
    modelEfficientNet_AU = EfficientNetModel_IFOF(image_input_dim=1280, feature_input_dim=35).to(device)

    # Define loss functions
    criterion_EH = nn.CrossEntropyLoss(weight=class_weights)
    criterion_AU = nn.CrossEntropyLoss(weight=class_weights)

    # Define optimizers
    optimizerResNet_EH = optim.Adam(modelResNet_EH.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizerResNet_AU = optim.Adam(modelResNet_AU.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizerEfficientNet_EH = optim.Adam(modelEfficientNet_EH.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizerEfficientNet_AU = optim.Adam(modelEfficientNet_AU.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train(data_loader, model, optimizer, criterion, device):
        model.train()
        total_loss = 0
        all_labels = []
        all_probs = []
        
        for images, features, labels in data_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images.unsqueeze(1), features.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
        return total_loss / len(data_loader), all_labels, all_probs


    # Evaluate function for IFOF model
    def evaluate(data_loader, model, criterion, device):
        model.eval()
        total_loss = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, features, labels in data_loader:
                images, features, labels = images.to(device), features.to(device), labels.to(device)
                outputs = model(images.unsqueeze(1), features.unsqueeze(1))
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return total_loss / len(data_loader), all_labels, all_probs

    # Train loop function with early stopping
    def trainloop(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, patience=5):
        best_val_loss = float('inf')
        no_improvement = 0
        best_model = None
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
                   'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': [],
                   'train_f1': [], 'val_f1': []}
        
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_labels, train_probs = train(train_loader, model, optimizer, criterion, device)
            val_loss, val_labels, val_probs = evaluate(val_loader, model, criterion, device)
            
            # Calculate metrics
            train_preds = np.argmax(train_probs, axis=1)
            val_preds = np.argmax(val_probs, axis=1)
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_accuracy'].append(val_accuracy)
            history['train_precision'].append(train_precision)
            history['val_precision'].append(val_precision)
            history['train_recall'].append(train_recall)
            history['val_recall'].append(val_recall)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, '
                f'Train Prec: {train_precision:.4f}, Val Prec: {val_precision:.4f}, '
                f'Train Rec: {train_recall:.4f}, Val Rec: {val_recall:.4f}, '
                f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
        
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f'Early stopping after epoch {epoch+1}')
                    break
        
        return best_model, history

    # Train and save models
    # modelResNet_EH_trained, historyResNet_EH = trainloop(train_loader_EH, val_loader_EH, modelResNet_EH, optimizerResNet_EH, criterion_EH, device, num_epochs)
    # modelResNet_AU_trained, historyResNet_AU = trainloop(train_loader_AU, val_loader_AU, modelResNet_AU, optimizerResNet_AU, criterion_AU, device, num_epochs)
    # modelEfficientNet_EH_trained, historyEfficientNet_EH = trainloop(train_loader_EH, val_loader_EH, modelEfficientNet_EH, optimizerEfficientNet_EH, criterion_EH, device, num_epochs)
    modelEfficientNet_AU_trained, historyEfficientNet_AU = trainloop(train_loader_AU, val_loader_AU, modelEfficientNet_AU, optimizerEfficientNet_AU, criterion_AU, device, num_epochs)

    # Set models and histories as user attributes in the study
    # study.set_user_attr('modelResNet_EH', modelResNet_EH_trained)
    # study.set_user_attr('modelResNet_AU', modelResNet_AU_trained)
    # study.set_user_attr('modelEfficientNet_EH', modelEfficientNet_EH_trained)
    study.set_user_attr('modelEfficientNet_AU', modelEfficientNet_AU_trained)

    # study.set_user_attr('historyResNet_EH', historyResNet_EH)
    # study.set_user_attr('historyResNet_AU', historyResNet_AU)
    # study.set_user_attr('historyEfficientNet_EH', historyEfficientNet_EH)
    study.set_user_attr('historyEfficientNet_AU', historyEfficientNet_AU)

    # Compute average validation loss or accuracy across models
    # val_loss_resnet_eh, _, _ = evaluate(val_loader_EH, modelResNet_EH_trained, criterion_EH, device)
    # val_loss_resnet_au, _, _ = evaluate(val_loader_AU, modelResNet_AU_trained, criterion_AU, device)
    # val_loss_efficientnet_eh, _, _ = evaluate(val_loader_EH, modelEfficientNet_EH_trained, criterion_EH, device)
    val_loss_efficientnet_au, _, _ = evaluate(val_loader_AU, modelEfficientNet_AU_trained, criterion_AU, device)

    # Average validation loss for optimization
    # val_loss_avg = (val_loss_resnet_eh + val_loss_resnet_au + val_loss_efficientnet_eh + val_loss_efficientnet_au) / 4.0
    val_loss_avg = val_loss_efficientnet_au
    return val_loss_avg  # Optuna minimizes this value

# Run the hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Save the study object to a file
with open('hyper-1/optuna_study_AU-2.pkl', 'wb') as f:
    pickle.dump(study, f)
# Print the best hyperparameters found
print('Best trial:')
trial = study.best_trial
print(f'  Loss: {trial.value:.4f}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Retrieve the best models and histories
# best_model_ResNet_EH = study.user_attrs['modelResNet_EH']
# best_model_ResNet_AU = study.user_attrs['modelResNet_AU']
# best_model_EfficientNet_EH = study.user_attrs['modelEfficientNet_EH']
best_model_EfficientNet_AU = study.user_attrs['modelEfficientNet_AU']

# best_history_ResNet_EH = study.user_attrs['historyResNet_EH']
# best_history_ResNet_AU = study.user_attrs['historyResNet_AU']
# best_history_EfficientNet_EH = study.user_attrs['historyEfficientNet_EH']
best_history_EfficientNet_AU = study.user_attrs['historyEfficientNet_AU']

# Save each best model to a .pth file
# torch.save(best_model_ResNet_EH.state_dict(), 'hyper-1/best_model_ResNet_EH.pth')
# torch.save(best_model_ResNet_AU.state_dict(), 'hyper-1/best_model_ResNet_AU.pth')
# torch.save(best_model_EfficientNet_EH.state_dict(), 'hyper-1/best_model_EfficientNet_EH.pth')
torch.save(best_model_EfficientNet_AU.state_dict(), 'hyper-1/best_model_EfficientNet_AU.pth')

# Save each best history to a .pkl file
# with open('hyper-1/best_history_ResNet_EH.pkl', 'wb') as f:
#     pickle.dump(best_history_ResNet_EH, f)

# with open('hyper-1/best_history_ResNet_AU.pkl', 'wb') as f:
#     pickle.dump(best_history_ResNet_AU, f)

# with open('hyper-1/best_history_EfficientNet_EH.pkl', 'wb') as f:
#     pickle.dump(best_history_EfficientNet_EH, f)

with open('hyper-1/best_history_EfficientNet_AU.pkl', 'wb') as f:
    pickle.dump(best_history_EfficientNet_AU, f)
