import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from .model import SoundClassifier, SoundClassifierUpdated, AudioClassifierWithTransformer
from .dataset import CustomDataModule
from .utils import test_model, train_one_epoch, validate_one_epoch
import copy

# Login to Weights & Biases
wandb.login()

# Common configuration for all experiments
common_config = {
    'num_epochs': 100,
    'k_folds': 5,
    'num_classes': 10,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'input_length': 16000,
    'num_workers': 2,
    'data_directory': '',  # Specify your data directory here
    'data_frame': None,  # Assign your DataFrame here
    'sampling_rate': 44100,
    'new_sampling_rate': 16000,
    'sample_length_seconds': 1,
    'file_column': 'filename',
    'label_column': 'category',
    'esc_10_flag': True
}

# Function to setup and return DataModule
def setup_data_module(config, fold):
    return CustomDataModule(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        data_directory=config['data_directory'],
        data_frame=config['data_frame'],
        validation_fold=fold,
        testing_fold=1,
        esc_10_flag=config['esc_10_flag'],
        file_column=config['file_column'],
        label_column=config['label_column'],
        sampling_rate=config['sampling_rate'],
        new_sampling_rate=config['new_sampling_rate'],
        sample_length_seconds=config['sample_length_seconds']
    )

#Architecture 1 : Before using dropout, early stopping and regularization
def arch1_exp1():
    config = common_config.copy()

    for fold in range(1, config['k_folds'] + 1):
        if fold == 1:
            continue

        wandb.init(project='audio_classification_assn2', entity='m23csa001', config=config, name=f'CNN Fold {fold}', reinit=True)

        data_module = setup_data_module(config, fold)
        data_module.setup(stage='fit', current_fold=fold)
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        data_module.setup(stage='test', current_fold=1)
        test_loader = data_module.test_dataloader()

        model = SoundClassifier(sequence_length=config['input_length'], output_size=config['num_classes']).to(config['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = trains_one_epoch(model_1, train_loader, criterion, optimizer, config['device'])
        val_loss, val_accuracy = validate_one_epoch(model_1, val_loader, criterion, config['device'])
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == config["num_epochs"]:
           print(f'Fold {fold}, Epoch [{epoch+1}/{config["num_epochs"]}] - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
           print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')

        test_loss, test_accuracy, test_f1, test_roc_auc = test_model(model_1, test_loader, criterion, config['device'], config['num_classes'])
        print(f'Fold {fold}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.2f}, ROC AUC: {test_roc_auc:.2f}')
        wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_f1': test_f1, 'test_roc_auc': test_roc_auc})

        wandb.finish()

#Architecture 1 : After using dropout, early stopping and regularization
def arch1_exp2():
    config = common_config.copy()
    config.update({
        'patience': 20,
    })

    for fold in range(1, config['k_folds'] + 1):
        if fold == 1:
            continue
        wandb.init(project='audio_classification_assn2', entity='m23csa001', config=config, name=f'CNN(Dr,ES,Re) Fold {fold}', reinit=True)

    data_module = CustomDataModule(batch_size=config['batch_size'],
                                   num_workers=config['num_workers'],
                                   data_directory=path,
                                   data_frame=df,
                                   validation_fold=fold,
                                   testing_fold=1,
                                   esc_10_flag=True,
                                   file_column='filename',
                                   label_column='category',
                                   sampling_rate=44100,
                                   new_sampling_rate=16000,
                                   sample_length_seconds=1
                                   )

    data_module.setup(stage='fit', current_fold=fold)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    data_module.setup(stage='test', current_fold=1)
    test_loader = data_module.test_dataloader()

    model_2 = SoundClassifierupdated(sequence_length=config['input_length'], output_size=config['num_classes']).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_2.parameters(), lr=config['learning_rate'])

    best_val_loss = float('inf')
    wait = 0
    best_model_wts = None

    for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = trains_one_epoch(model_2, train_loader, criterion, optimizer, config['device'])
        val_loss, val_accuracy = validate_one_epoch(model_2, val_loader, criterion, config['device'])
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            best_model_wts = copy.deepcopy(model_2.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping on fold {fold}, epoch {epoch+1}")
                break

        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == config["num_epochs"]:
            print(f'Fold {fold}, Epoch [{epoch+1}/{config["num_epochs"]}] - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')

    if best_model_wts:
        model_2.load_state_dict(best_model_wts)

    test_loss, test_accuracy, test_f1, test_roc_auc = test_model(model_2, test_loader, criterion, config['device'], config['num_classes'])
    print(f'Fold {fold}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.2f}, ROC AUC: {test_roc_auc:.2f}')
    wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_f1': test_f1, 'test_roc_auc': test_roc_auc})

    wandb.finish()

#Architecture2 :

#Exp 1 : Num heads = 1

def arch2_exp1():
    config = common_config.copy()
    config.update({
        'num_heads': 1,
    })

   for fold in range(1, config['k_folds'] + 1):
     if fold == 1:
        continue

    wandb.init(project='audio_classification_assn2', entity='m23csa001',config=config, name=f'NH = 1,Fold {fold}', reinit=True)

    data_module = CustomDataModule(batch_size=config['batch_size'],
                                   num_workers=config['num_workers'],
                                   data_directory=path,
                                   data_frame=df,
                                   validation_fold=fold,
                                   testing_fold=1,
                                   esc_10_flag=True,
                                   file_column='filename',
                                   label_column='category',
                                   sampling_rate=44100,
                                   new_sampling_rate=16000,
                                   sample_length_seconds=1
                                   )

    data_module.setup(stage='fit', current_fold=fold)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    data_module.setup(stage='test', current_fold=1)
    test_loader = data_module.test_dataloader()

    model_6 = AudioClassifierWithTransformer(num_classes=config['num_classes'], input_length=config['input_length'],
                                             embed_size=config['embed_size'], num_heads=config['num_heads'], num_encoder_layers=config['num_encoder_layers']).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_6.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = train_one_epoch(model_6, train_loader, criterion, optimizer, config['device'], max_norm=config['max_norm'])
        val_loss, val_accuracy = validate_one_epoch(model_6, val_loader, criterion, config['device'])
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == config["num_epochs"]:
           print(f'Fold {fold}, Epoch [{epoch+1}/{config["num_epochs"]}] - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
           print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')

    test_loss, test_accuracy, test_f1, test_roc_auc = test_model(model_6, test_loader, criterion, config['device'], config['num_classes'])
    print(f'Fold {fold}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.2f}, ROC AUC: {test_roc_auc:.2f}')
    wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_f1': test_f1, 'test_roc_auc': test_roc_auc})

    wandb.finish()

#Num head = 2
def arch2_exp2():
    config = common_config.copy()
    config.update({
        'num_heads': 2,
    })
  
    for fold in range(1, config['k_folds'] + 1):
      if fold == 1:
        continue

    wandb.init(project='audio_classification_assn2', entity='m23csa001',config=config, name=f'NH = 1,Fold {fold}', reinit=True)

    data_module = CustomDataModule(batch_size=config['batch_size'],
                                   num_workers=config['num_workers'],
                                   data_directory=path,
                                   data_frame=df,
                                   validation_fold=fold,
                                   testing_fold=1,
                                   esc_10_flag=True,
                                   file_column='filename',
                                   label_column='category',
                                   sampling_rate=44100,
                                   new_sampling_rate=16000,
                                   sample_length_seconds=1
                                   )

    data_module.setup(stage='fit', current_fold=fold)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    data_module.setup(stage='test', current_fold=1)
    test_loader = data_module.test_dataloader()

    model_5 = AudioClassifierWithTransformer(num_classes=config['num_classes'], input_length=config['input_length'],
                                             embed_size=config['embed_size'], num_heads=config['num_heads'], num_encoder_layers=config['num_encoder_layers']).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_5.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = train_one_epoch(model_5, train_loader, criterion, optimizer, config['device'], max_norm=config['max_norm'])
        val_loss, val_accuracy = validate_one_epoch(model_5, val_loader, criterion, config['device'])
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == config["num_epochs"]:
           print(f'Fold {fold}, Epoch [{epoch+1}/{config["num_epochs"]}] - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
           print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')

    test_loss, test_accuracy, test_f1, test_roc_auc = test_model(model_5, test_loader, criterion, config['device'], config['num_classes'])
    print(f'Fold {fold}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.2f}, ROC AUC: {test_roc_auc:.2f}')
    wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_f1': test_f1, 'test_roc_auc': test_roc_auc})

    wandb.finish()

#Num head = 4
def arch2_exp3():
    config = common_config.copy()
    config.update({
        'num_heads': 2,
    })

   for fold in range(1, config['k_folds'] + 1):
     if fold == 1:
       continue

    wandb.init(project='audio_classification_assn2', entity='m23csa001',config=config, name=f'NH = 1,Fold {fold}', reinit=True)

    data_module = CustomDataModule(batch_size=config['batch_size'],
                                   num_workers=config['num_workers'],
                                   data_directory=path,
                                   data_frame=df,
                                   validation_fold=fold,
                                   testing_fold=1,
                                   esc_10_flag=True,
                                   file_column='filename',
                                   label_column='category',
                                   sampling_rate=44100,
                                   new_sampling_rate=16000,
                                   sample_length_seconds=1
                                   )
    data_module.setup(stage='fit', current_fold=fold)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    data_module.setup(stage='test', current_fold=1)
    test_loader = data_module.test_dataloader()

    model_7 = AudioClassifierWithTransformer(num_classes=config['num_classes'], input_length=config['input_length'],
                                             embed_size=config['embed_size'], num_heads=config['num_heads'], num_encoder_layers=config['num_encoder_layers']).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_7.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = train_one_epoch(model_7, train_loader, criterion, optimizer, config['device'], max_norm=config['max_norm'])
        val_loss, val_accuracy = validate_one_epoch(model_7, val_loader, criterion, config['device'])
        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == config["num_epochs"]:
           print(f'Fold {fold}, Epoch [{epoch+1}/{config["num_epochs"]}] - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
           print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')

    test_loss, test_accuracy, test_f1, test_roc_auc = test_model(model_7, test_loader, criterion, config['device'], config['num_classes'])
    print(f'Fold {fold}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.2f}, ROC AUC: {test_roc_auc:.2f}')
    wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_f1': test_f1, 'test_roc_auc': test_roc_auc})

    wandb.finish()

#Experiment 4 : Max norm = 1.5 , learning rate scheduler and optimizer as AdamW
def arch2_exp4():
    config = common_config.copy()
    config.update({
        'max_norm': 1.5,
        'num_heads': 2    
    })


    for fold in range(1, config['k_folds'] + 1):
      if fold == 1:
        continue

    wandb.init(project='audio_classification_assn2', entity='m23csa001', config=config, name=f'NH = 1, Fold {fold}', reinit=True)

    data_module = CustomDataModule(batch_size=config['batch_size'],
                                   num_workers=config['num_workers'],
                                   data_directory=path,
                                   data_frame=df,
                                   validation_fold=fold,
                                   testing_fold=1,
                                   esc_10_flag=True,
                                   file_column='filename',
                                   label_column='category',
                                   sampling_rate=44100,
                                   new_sampling_rate=16000,
                                   sample_length_seconds=1
                                   )
    data_module.setup(stage='fit', current_fold=fold)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    data_module.setup(stage='test', current_fold=1)
    test_loader = data_module.test_dataloader()

    model_8 = AudioClassifierWithTransformer(num_classes=config['num_classes'], input_length=config['input_length'],
                                             embed_size=config['embed_size'], num_heads=config['num_heads'], num_encoder_layers=config['num_encoder_layers']).to(config['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_8.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])  # Changed to AdamW

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    for epoch in range(config['num_epochs']):
        train_loss, train_accuracy = train_one_epoch(model_8, train_loader, criterion, optimizer, config['device'], max_norm=config['max_norm'])
        val_loss, val_accuracy = validate_one_epoch(model_8, val_loader, criterion, config['device'])

        scheduler.step(val_loss)

        wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'lr': optimizer.param_groups[0]['lr']})

        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == config["num_epochs"]:
            print(f'Fold {fold}, Epoch [{epoch+1}/{config["num_epochs"]}] - Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n')


    test_loss, test_accuracy, test_f1, test_roc_auc = test_model(model_8, test_loader, criterion, config['device'], config['num_classes'])
    print(f'Fold {fold}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {test_f1:.2f}, ROC AUC: {test_roc_auc:.2f}')
    wandb.log({'test_loss': test_loss, 'test_accuracy': test_accuracy, 'test_f1': test_f1, 'test_roc_auc': test_roc_auc})

    wandb.finish()
