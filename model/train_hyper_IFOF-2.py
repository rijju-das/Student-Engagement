import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from data_IF import get_image_dataloaders
from data_IFOF import get_dataloaders, get_class_weights
from model import ResNetModel_IFOF, ResNetModel_IF, EfficientNetModel_IF, EfficientNetModel_IFOF

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
    train_loader_IF, val_loader_IF, _ = get_image_dataloaders(image_dir, batch_size)
    train_loader_IFOF, val_loader_IFOF, _ = get_dataloaders(image_dir, feature_dir, batch_size)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get class weights for IFOF model and pass them to the loss function
    train_labels = np.array([train_loader_IFOF.dataset.dataset.labels[i] for i in train_loader_IFOF.dataset.indices])
    class_weights = torch.tensor(get_class_weights(train_labels)).float().to(device)

    # Define loss functions
    criterion_IF = nn.CrossEntropyLoss(weight=class_weights)  
    criterion_IFOF = nn.CrossEntropyLoss(weight=class_weights)

    # Initialize models
    modelResNet_IF = ResNetModel_IF().to(device)
    modelResNet_IFOF = ResNetModel_IFOF().to(device)
    modelEfficientNet_IF = EfficientNetModel_IF().to(device)
    modelEfficientNet_IFOF = EfficientNetModel_IFOF().to(device)

    # Define optimizers
    optimizerResNet_IF = optim.Adam(modelResNet_IF.parameters(), lr=learning_rate)
    optimizerResNet_IFOF = optim.Adam(modelResNet_IFOF.parameters(), lr=learning_rate)
    optimizerEfficientNet_IF = optim.Adam(modelEfficientNet_IF.parameters(), lr=learning_rate)
    optimizerEfficientNet_IFOF = optim.Adam(modelEfficientNet_IFOF.parameters(), lr=learning_rate)


    # Train function for IF model
    def train_IF(data_loader, model, optimizer, criterion, device):
        model.train()
        total_loss = 0
        all_labels = []
        all_probs = []
        
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
        return total_loss / len(data_loader), all_labels, all_probs

    # Train function for IFOF model
    def train_IFOF(data_loader, model, optimizer, criterion, device):
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

    # Evaluate function for IF model
    def evaluate_IF(data_loader, model, criterion, device):
        model.eval()
        total_loss = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return total_loss / len(data_loader), all_labels, all_probs

    # Evaluate function for IFOF model
    def evaluate_IFOF(data_loader, model, criterion, device):
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

    # Train loop function for IF model
    def trainloop_IF(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, patience=5):
        best_val_loss = float('inf')
        no_improvement = 0
        best_model = None
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
                   'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': [],
                   'train_f1': [], 'val_f1': []}
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_labels, train_probs = train_IF(train_loader, model, optimizer, criterion, device)
            val_loss, val_labels, val_probs = evaluate_IF(val_loader, model, criterion, device)
            
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

    # Train loop function for IFOF model
    def trainloop_IFOF(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, patience=5):
        best_val_loss = float('inf')
        no_improvement = 0
        best_model = None
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
                   'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': [],
                   'train_f1': [], 'val_f1': []}
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_labels, train_probs = train_IFOF(train_loader, model, optimizer, criterion, device)
            val_loss, val_labels, val_probs = evaluate_IFOF(val_loader, model, criterion, device)
            
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

    # Training and saving the models and history
    # modelResNet_IF_trained, historyResNet_IF = trainloop_IF(train_loader_IF, val_loader_IF, modelResNet_IF, optimizerResNet_IF, criterion_IF, device, num_epochs)
    # modelResNet_IFOF_trained, historyResNet_IFOF = trainloop_IFOF(train_loader_IFOF, val_loader_IFOF, modelResNet_IFOF, optimizerResNet_IFOF, criterion_IFOF, device, num_epochs)
    # modelEfficientNet_IF_trained, historyEfficientNet_IF = trainloop_IF(train_loader_IF, val_loader_IF, modelEfficientNet_IF, optimizerEfficientNet_IF, criterion_IF, device, num_epochs)
    modelEfficientNet_IFOF_trained, historyEfficientNet_IFOF = trainloop_IFOF(train_loader_IFOF, val_loader_IFOF, modelEfficientNet_IFOF, optimizerEfficientNet_IFOF, criterion_IFOF, device, num_epochs)

    # Set models and histories as user attributes in the study
    # study.set_user_attr('modelResNet_IF', modelResNet_IF_trained)
    # study.set_user_attr('modelResNet_IFOF', modelResNet_IFOF_trained)
    # study.set_user_attr('modelEfficientNet_IF', modelEfficientNet_IF_trained)
    study.set_user_attr('modelEfficientNet_IFOF', modelEfficientNet_IFOF_trained)

    # study.set_user_attr('historyResNet_IF', historyResNet_IF)
    # study.set_user_attr('historyResNet_IFOF', historyResNet_IFOF)
    # study.set_user_attr('historyEfficientNet_IF', historyEfficientNet_IF)
    study.set_user_attr('historyEfficientNet_IFOF', historyEfficientNet_IFOF)

    # Compute average validation loss or accuracy across models
    # val_loss_resnet_IF, _, _ = evaluate_IF(val_loader_IF, modelResNet_IF_trained, criterion_IF, device)
    # val_loss_resnet_IFOF, _, _ = evaluate_IFOF(val_loader_IFOF, modelResNet_IFOF_trained, criterion_IFOF, device)
    # val_loss_efficientnet_IF, _, _ = evaluate_IF(val_loader_IF, modelEfficientNet_IF_trained, criterion_IF, device)
    val_loss_efficientnet_IFOF, _, _ = evaluate_IFOF(val_loader_IFOF, modelEfficientNet_IFOF_trained, criterion_IFOF, device)

    # Average validation loss for optimization
    # val_loss_avg = (val_loss_resnet_IF + val_loss_resnet_IFOF + val_loss_efficientnet_IF + val_loss_efficientnet_IFOF) / 4.0
    val_loss_avg = val_loss_efficientnet_IFOF
    return val_loss_avg  # Optuna minimizes this value

# Run the hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Save the study object to a file
with open('hyper-1/optuna_study_IFOF-2.pkl', 'wb') as f:
    pickle.dump(study, f)
# Print the best hyperparameters found
print('Best trial:')
trial = study.best_trial
print(f'  Loss: {trial.value:.4f}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# Retrieve the best models and histories
# best_model_ResNet_IF = study.user_attrs['modelResNet_IF']
# best_model_ResNet_IFOF = study.user_attrs['modelResNet_IFOF']
# best_model_EfficientNet_IF = study.user_attrs['modelEfficientNet_IF']
best_model_EfficientNet_IFOF = study.user_attrs['modelEfficientNet_IFOF']

# best_history_ResNet_IF = study.user_attrs['historyResNet_IF']
# best_history_ResNet_IFOF = study.user_attrs['historyResNet_IFOF']
# best_history_EfficientNet_IF = study.user_attrs['historyEfficientNet_IF']
best_history_EfficientNet_IFOF = study.user_attrs['historyEfficientNet_IFOF']

# Save each best model to a .pth file
# torch.save(best_model_ResNet_IF.state_dict(), 'hyper-1/best_model_ResNet_IF.pth')
# torch.save(best_model_ResNet_IFOF.state_dict(), 'hyper-1/best_model_ResNet_IFOF.pth')
# torch.save(best_model_EfficientNet_IF.state_dict(), 'hyper-1/best_model_EfficientNet_IF.pth')
torch.save(best_model_EfficientNet_IFOF.state_dict(), 'hyper-1/best_model_EfficientNet_IFOF.pth')

# Save each best history to a .pkl file
# with open('hyper-1/best_history_ResNet_IF.pkl', 'wb') as f:
#     pickle.dump(best_history_ResNet_IF, f)

# with open('hyper-1/best_history_ResNet_IFOF.pkl', 'wb') as f:
#     pickle.dump(best_history_ResNet_IFOF, f)

# with open('hyper-1/best_history_EfficientNet_IF.pkl', 'wb') as f:
#     pickle.dump(best_history_EfficientNet_IF, f)

with open('hyper-1/best_history_EfficientNet_IFOF.pkl', 'wb') as f:
    pickle.dump(best_history_EfficientNet_IFOF, f)
