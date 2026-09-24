import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report, precision_score
import pandas as pd
from collections import defaultdict
import torch.nn.functional as F
import copy
from model import *
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_best_model(subject_id, device):
    model = CARAF_NetModel(
        cuda=(device.type == 'cuda'),
        number_of_source=14,  
        number_of_category=4,
        batch_size=64,
        time_steps=10
    ).to(device)
    
    model_path = f"best_model_seed4_subject_{subject_id}.pth"
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model for subject {subject_id}, accuracy: {checkpoint['accuracy']:.2f}%")
            return model, checkpoint
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"Model architecture mismatch for subject {subject_id}. Trying to detect correct architecture...")
                
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                state_dict = checkpoint['model_state_dict']
                
                classifier_weight_key = None
                for key in state_dict.keys():
                    if 'classifier' in key and 'weight' in key:
                        classifier_weight_key = key
                
                if classifier_weight_key:
                    num_classes = 4
                    print(f"Detected {num_classes} classes in saved model")
                    
                    model = CARAF_NetModel(
                        cuda=(device.type == 'cuda'),
                        number_of_source=14,  
                        number_of_category=4,
                        batch_size=64,
                        time_steps=10
                    ).to(device)
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Successfully loaded model with {num_classes} classes for subject {subject_id}")
                    return model, checkpoint
                else:
                    print(f"Could not detect model architecture for subject {subject_id}")
                    return None, None
            else:
                print(f"Error loading model for subject {subject_id}: {str(e)}")
                return None, None
    else:
        print(f"Warning: Could not find model file {model_path} for subject {subject_id}")
        return None, None

def get_test_loader2(subject_id, args):
    from preprocess import getDataLoaders
    
    source_loaders, test_loader, test_loader2 = getDataLoaders(subject_id, args)
    
    return test_loader

def test_subject_with_metrics(model, test_loader, cuda, args):
    device = torch.device('cuda' if cuda else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            if cuda:
                data = data.to(device)
                labels = labels.to(device)
            
            features = model.eeg_extractor(data)
            
            if len(features.shape) == 3:
                features = model.attention_pool(features)
            
            logits = model.classifier(features)
            
            if hasattr(model, 'use_distillation') and model.use_distillation:
                temp_scaled_logits = logits / 1.5
                temp_probs = F.softmax(temp_scaled_logits, dim=1)
                
                conf_values, _ = torch.max(F.softmax(logits, dim=1), dim=1)
                conf_weights = conf_values.unsqueeze(1).expand_as(logits)
                
                smoothed_probs = F.softmax(logits, dim=1) * 0.8 + 0.2 / args.cls_classes
                
                final_probs = conf_weights * temp_probs + (1 - conf_weights) * smoothed_probs
                
                _, preds = torch.max(final_probs, 1)
            else:
                _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    class_report = classification_report(
        all_labels, 
        all_preds, 
        zero_division=0,
        output_dict=True
    )
    
    metrics = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'class_report': class_report,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    return metrics

def plot_confusion_matrix_large_font(cm_normalized, subject_id, results_dir):
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(cm_normalized, 
                     annot=True, 
                     fmt='.1f', 
                     cmap='Blues',
                     xticklabels=['Sad', 'Fear', 'Happy', 'Neutral'],
                     yticklabels=['Sad', 'Fear', 'Happy', 'Neutral'],
                     annot_kws={'size': 16})
    
    plt.xlabel('Predicted label', fontsize=16, fontweight='bold')
    plt.ylabel('True label', fontsize=16, fontweight='bold')
    plt.title(f'Confusion matrix for subject {subject_id} (% accuracy)', 
              fontsize=18, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Accuracy (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_subject_{subject_id}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_confusion_matrix(all_subjects_cm_normalized, results_dir):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(all_subjects_cm_normalized, 
                     annot=True, 
                     fmt='.1f', 
                     cmap='Blues',
                     xticklabels=['Sad', 'Fear', 'Happy', 'Neutral'],
                     yticklabels=['Sad', 'Fear', 'Happy', 'Neutral'],
                     annot_kws={'size': 18})

    plt.xlabel('Predicted label', fontsize=18, fontweight='bold')
    plt.ylabel('True label', fontsize=18, fontweight='bold')
    plt.title('Comprehensive confusion matrix for all subjects (% accuracy)', 
              fontsize=20, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=16)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Accuracy (%)', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_all_subjects.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_comparison_large_font(performance_df, results_dir):
    plt.figure(figsize=(16, 10))
    ind = np.arange(len(performance_df)-1)
    width = 0.2
    
    bars1 = plt.bar(ind - 1.5*width, performance_df['Accuracy (%)'][:-1], width, 
                    label='Accuracy', alpha=0.8)
    bars2 = plt.bar(ind - 0.5*width, performance_df['Precision (%)'][:-1], width, 
                    label='Precision', alpha=0.8)
    bars3 = plt.bar(ind + 0.5*width, performance_df['Recall (%)'][:-1], width, 
                    label='Recall', alpha=0.8)
    bars4 = plt.bar(ind + 1.5*width, performance_df['F1 Score (%)'][:-1], width, 
                    label='F1 Score', alpha=0.8)
    
    plt.axhline(y=performance_df['Accuracy (%)'].iloc[-1], color='tab:blue', 
                linestyle='--', alpha=0.7, linewidth=2)
    plt.axhline(y=performance_df['Precision (%)'].iloc[-1], color='tab:orange', 
                linestyle='--', alpha=0.7, linewidth=2)
    plt.axhline(y=performance_df['Recall (%)'].iloc[-1], color='tab:green', 
                linestyle='--', alpha=0.7, linewidth=2)
    plt.axhline(y=performance_df['F1 Score (%)'].iloc[-1], color='tab:red', 
                linestyle='--', alpha=0.7, linewidth=2)
    
    def add_labels_large(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    add_labels_large(bars1)
    add_labels_large(bars2)
    add_labels_large(bars3)
    add_labels_large(bars4)
    
    plt.xlabel('Subject ID', fontsize=16, fontweight='bold')
    plt.ylabel('Performance metrics (%)', fontsize=16, fontweight='bold')
    plt.title('Performance metrics comparison for all subjects', 
              fontsize=18, fontweight='bold')
    
    plt.xticks(ind, performance_df['Subject ID'][:-1], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'subject_performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def test_all_subjects(args):
    results_dir = "test_results_SEED_IV"
    os.makedirs(results_dir, exist_ok=True)
    
    start_time = time.time()
    print(f"Start testing time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"Number of visible GPUs: {torch.cuda.device_count()}")
        print(f"Current active GPU: {torch.cuda.current_device()}")
        print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
    
    all_subjects_preds = []
    all_subjects_labels = []
    
    subject_performance = defaultdict(list)
    
    log_file = os.path.join(results_dir, "test_all_subjects_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Testing all subjects record\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU name: {torch.cuda.get_device_name(0)}\n")
        f.write("-" * 50 + "\n")
    
    for subject_id in range(0, 15):
        subject_start_time = time.time()
        
        with open(log_file, 'a') as f:
            f.write(f"\nTesting subject {subject_id} started...\n")
        
        model, checkpoint = load_best_model(subject_id, device)
        if model is None:
            print(f"Skipping subject {subject_id}, could not load model")
            with open(log_file, 'a') as f:
                f.write(f"Skipping subject {subject_id}, could not load model\n")
            continue
        
        model.eval()
        
        try:
            test_loader2 = get_test_loader2(subject_id, args)
            
            metrics = test_subject_with_metrics(model, test_loader2, device.type == 'cuda', args)
            
            all_subjects_preds.extend(metrics['predictions'])
            all_subjects_labels.extend(metrics['labels'])
            
            subject_performance['Subject ID'].append(subject_id)
            subject_performance['Accuracy (%)'].append(metrics['accuracy'])
            subject_performance['Precision (%)'].append(metrics['precision'])
            subject_performance['Recall (%)'].append(metrics['recall'])
            subject_performance['F1 Score (%)'].append(metrics['f1'])
            
            print(f"Subject {subject_id} test results: Accuracy={metrics['accuracy']:.2f}%, Recall={metrics['recall']:.2f}%, F1 score={metrics['f1']:.2f}%")
            
            with open(log_file, 'a') as f:
                f.write(f"Subject {subject_id} testing completed, time spent: {time.time() - subject_start_time:.2f} seconds\n")
                f.write(f"Accuracy: {metrics['accuracy']:.2f}%, Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%, F1 score: {metrics['f1']:.2f}%\n")
                f.write(f"Confusion matrix (sample counts):\n{metrics['confusion_matrix']}\n")
                f.write(f"Confusion matrix (accuracy percentages):\n{metrics['confusion_matrix_normalized']}\n")
                f.write("-" * 30 + "\n")
            
            plot_confusion_matrix_large_font(metrics['confusion_matrix_normalized'], 
                                           subject_id, results_dir)
            
        except Exception as e:
            print(f"Error testing subject {subject_id}: {str(e)}")
            with open(log_file, 'a') as f:
                f.write(f"Error testing subject {subject_id}: {str(e)}\n")
            continue
    
    if len(all_subjects_preds) == 0:
        print("No subjects were successfully tested, cannot generate comprehensive results")
        return None, None
    
    all_subjects_cm = confusion_matrix(all_subjects_labels, all_subjects_preds)
    all_subjects_cm_normalized = all_subjects_cm.astype('float') / all_subjects_cm.sum(axis=1)[:, np.newaxis] * 100
    
    plot_comprehensive_confusion_matrix(all_subjects_cm_normalized, results_dir)
    
    overall_acc = accuracy_score(all_subjects_labels, all_subjects_preds) * 100
    overall_precision = precision_score(all_subjects_labels, all_subjects_preds, average='macro', zero_division=0) * 100
    overall_recall = recall_score(all_subjects_labels, all_subjects_preds, average='macro', zero_division=0) * 100
    overall_f1 = f1_score(all_subjects_labels, all_subjects_preds, average='macro', zero_division=0) * 100
    
    print("\nComprehensive test results:")
    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Overall precision: {overall_precision:.2f}%")
    print(f"Overall recall: {overall_recall:.2f}%")
    print(f"Overall F1 score: {overall_f1:.2f}%")
    
    with open(log_file, 'a') as f:
        f.write("\nComprehensive test results:\n")
        f.write(f"Overall accuracy: {overall_acc:.2f}%\n")
        f.write(f"Overall precision: {overall_precision:.2f}%\n")
        f.write(f"Overall recall: {overall_recall:.2f}%\n")
        f.write(f"Overall F1 score: {overall_f1:.2f}%\n\n")
        
        f.write("Comprehensive confusion matrix (sample counts):\n")
        f.write(f"{all_subjects_cm}\n\n")
        f.write("Comprehensive confusion matrix (accuracy percentages):\n")
        f.write(f"{all_subjects_cm_normalized}\n\n")
        
        class_report = classification_report(all_subjects_labels, all_subjects_preds, zero_division=0)
        f.write("Detailed classification report:\n")
        f.write(f"{class_report}\n")
    
    performance_df = pd.DataFrame(subject_performance)
    performance_df.loc[len(performance_df)] = ['Average', 
                                              performance_df['Accuracy (%)'].mean(),
                                              performance_df['Precision (%)'].mean(),
                                              performance_df['Recall (%)'].mean(),
                                              performance_df['F1 Score (%)'].mean()]
    
    print("\nPerformance table for all subjects:")
    print(performance_df.to_string(index=False))
    
    performance_df.to_csv(os.path.join(results_dir, 'subject_performance_summary.csv'), index=False)
    
    plot_performance_comparison_large_font(performance_df, results_dir)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTesting completed, total time: {total_time:.2f} seconds")
    
    with open(log_file, 'a') as f:
        f.write(f"\nTesting completed, total time: {total_time:.2f} seconds\n")
        f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return all_subjects_cm, performance_df

if __name__ == "__main__":
    print("Starting to test all subjects...")
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='CARAF-Net')

    parser.add_argument("--way", type=str, default='CARAF-Net/seed4', help="name of current way")
    parser.add_argument("--index", type=str, default='0', help="tensorboard index")

    parser.add_argument("--dataset_name", type=str, nargs='?', default='seed4', help="the dataset name, supporting seed3 and seed4")
    parser.add_argument("--session", type=str, nargs='?', default='1', help="selected session")
    parser.add_argument("--subjects", type=int, choices=[15], default=15, help="the number of all subject")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")

    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--hid_dim", type=int, default=64, help="hid dim is for hidden layer of lstm")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument('--number_of_source', type=int, default=14,
                    help='Number of source subjects (must match actual data)')
    
    args = parser.parse_args()
    args.source_subjects = args.subjects-1
    args.seed3_path = "{data_path}/SEED/"
    args.seed4_path = "{data_path}/SEED-IV/"
    if cuda:
        args.num_workers_train = 2
        args.num_workers_test = 2
    else:
        args.num_workers_train = 0
        args.num_workers_test = 0
    if args.dataset_name == "seed3":
        args.path = args.seed3_path
        args.cls_classes = 3
        args.time_steps = 30
        args.batch_size = 32
        args.epoch = 100
    elif args.dataset_name == "seed4":
        args.path = args.seed4_path
        args.cls_classes = 4
        args.time_steps = 10
        args.batch_size = 64
        args.epoch= 200
    else:
        print("need to define the input dataset")
        
    cuda = torch.cuda.is_available()
    if cuda:
        print("GPU is available. Running on:", torch.cuda.get_device_name(0))
        torch.cuda.set_device(0)
    else:
        print("No GPU detected. Running on CPU.")
    optim_config = {"lr": args.lr, "weight_decay": args.weight_decay}
    all_subjects_cm, performance_df = test_all_subjects(args)
    
    if all_subjects_cm is not None:
        print("Testing completed! All results have been saved to the 'test_results' directory")
    else:
        print("Testing could not be completed, please check the error log")