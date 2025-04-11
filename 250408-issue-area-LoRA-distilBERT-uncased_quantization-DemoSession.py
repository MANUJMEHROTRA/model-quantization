# %%
import os
import json
import torch
import transformers
import accelerate
import huggingface_hub
import peft
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
from sklearn.utils import resample
from collections import Counter
import time
import onnxruntime as ort
import pickle

print("peft:", peft.__version__)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("Huggingface Hub:", huggingface_hub.__version__)


# %%
# PyTorch Quantization modules
import torch.ao.quantization
from torch.ao.quantization import quantize_dynamic, quantize, float_qparams_weight_only_qconfig
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.qconfig import QConfig
from torch.ao.quantization import prepare, convert, QConfigMapping, default_dynamic_qconfig, default_qconfig
import copy

# %%
# Device Selection Function
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# Device Configuration
device = get_device()

# %%
def update_model_dict(model_alias, MODEL_NAME):
    if not os.path.exists('model_dict.json'):
        model_dict = {}
    else:
        with open('model_dict.json', 'r') as file:
            model_dict = json.load(file)

    model_dict[model_alias] = MODEL_NAME

    with open('model_dict.json', 'w') as file:
        json.dump(model_dict, file)

# %%
def load_and_preprocess_data(filepath="./data/train-00000-of-00001-a5a7c6e4bb30b016.parquet", model_alias=""):
    """Loads and preprocesses the dataset."""
    df = pd.read_parquet(filepath)
    df = df[['conversation', 'issue_area']]
    print("Original distribution:\n", df['issue_area'].value_counts())
    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["issue_area"])

    #saving Label-encoder
    label_encoder_path = f"model-metric/{model_alias}/label_encoder.pkl"
    os.makedirs(os.path.dirname(label_encoder_path), exist_ok=True)
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
        
    return df, label_encoder

# %%
def balance_dataset(df, max_count=100, random_state=42):
    """Balances the dataset using oversampling."""
    balanced_df = pd.DataFrame()
    for issue in df['issue_area'].unique():
        subset = df[df['issue_area'] == issue]
        balanced_subset = resample(subset, replace=True, n_samples=max_count, random_state=random_state)
        balanced_df = pd.concat([balanced_df, balanced_subset])
    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

# %%
def preprocess_conversation(conversation):
    """Preprocesses a conversation."""
    if isinstance(conversation, list):
        return " ".join([turn.get('text', '') for turn in conversation if isinstance(turn, dict)])
    return str(conversation).lower()

# %%
# Define PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        inputs = self.tokenizer(
            row["conversation"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        label = torch.tensor(row["labels"], dtype=torch.long)
        return input_ids, attention_mask, label

# %%
def create_dataloaders(df, tokenizer, batch_size=8, train_ratio=0.75):
    """Creates train and test DataLoaders."""
    train_size = int(train_ratio * len(df))
    train_df, test_df = df[:train_size], df[train_size:]
    train_dataset = CustomDataset(train_df, tokenizer)
    test_dataset = CustomDataset(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_df

# %%
class DistilBERTWithLoRA(nn.Module):
    def __init__(self, num_labels, lora_r=4, lora_alpha=16, lora_dropout=0.1):
        super(DistilBERTWithLoRA, self).__init__()
        # Load the base model with the correct number of labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=num_labels  # Ensure this matches the number of classes
        )
        
        # LoRA Configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "k_lin", "v_lin"]
        )
        self.bert = get_peft_model(self.bert, lora_config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Return the logits directly

# %%
# Function to compute class weights
def compute_class_weights(labels, num_classes):
    counter = Counter(labels)
    total_samples = len(labels)
    weights = [total_samples / (num_classes * counter[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)

# %%
def train_model(model, train_loader, model_alias, epochs=3, learning_rate=5e-5, class_weights=None):
    """Trains the model and saves logs, metrics, and model weights."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Create directory for storing model metrics
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard writer in the model directory
    writer = SummaryWriter(log_dir=model_dir)

    # Set up loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epoch_losses = []
    metrics_data = []

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            labels = labels.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Log batch loss every 10 batches
            if batch_idx % 10 == 0:
                writer.add_scalar("BatchLoss/train", loss.item(), epoch * len(train_loader) + batch_idx)

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        epoch_time = time.time() - start_time

        # Store metrics for CSV logging
        metrics_data.append([epoch + 1, avg_loss, accuracy, precision, recall, f1, epoch_time])

        # Print metrics
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}, Time={epoch_time:.2f}s")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        writer.add_scalar("Precision/train", precision, epoch)
        writer.add_scalar("Recall/train", recall, epoch)
        writer.add_scalar("F1-score/train", f1, epoch)
        writer.add_scalar("Time/Epoch", epoch_time, epoch)

    # Save model KPIs as CSV
    metrics_df = pd.DataFrame(metrics_data, columns=["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"])
    metrics_df.to_csv(os.path.join(model_dir, "training_metrics.csv"), index=False)

    # Save training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(model_dir, "training_loss.png")
    plt.savefig(loss_plot_path)
    writer.add_figure("Training Loss", plt.gcf(), close=True)

    # Save model weights
    model_path = os.path.join(model_dir, f"{model_alias}.pth")
    torch.save(model.state_dict(), model_path)

    writer.flush()
    writer.close()

# %%
def evaluate_model(model, test_loader, label_encoder, model_alias):
    """Evaluates the model and saves metrics, logs, and confusion matrix."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create directory for storing model metrics
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=model_dir)

    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            labels = labels.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    eval_time = time.time() - start_time
    class_names = label_encoder.classes_

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })

    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Print and save classification report
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save confusion matrix plot
    confusion_matrix_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    writer.add_figure("Confusion Matrix", plt.gcf(), close=True)

    # Print overall metrics
    print("\nPer-class Metrics:\n", class_metrics.to_string(index=False))
    print(f"\nOverall Metrics:\nPrecision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1-score: {overall_f1:.4f}, Eval Time: {eval_time:.2f}s")

    # Log metrics to TensorBoard
    writer.add_scalar("Precision/test", overall_precision)
    writer.add_scalar("Recall/test", overall_recall)
    writer.add_scalar("F1-score/test", overall_f1)
    writer.add_scalar("Evaluation Time", eval_time)

    # Log per-class metrics
    for i, class_name in enumerate(class_names):
        writer.add_scalar(f"Precision/{class_name}", precision[i])
        writer.add_scalar(f"Recall/{class_name}", recall[i])
        writer.add_scalar(f"F1-score/{class_name}", f1[i])

    writer.flush()
    writer.close()

    # Save evaluation metrics
    class_metrics.to_csv(os.path.join(model_dir, "class_metrics.csv"), index=False)
    cm_df.to_csv(os.path.join(model_dir, "confusion_matrix.csv"))

    return overall_precision, overall_recall, overall_f1, eval_time, class_metrics, cm_df

# %%
def export_to_onnx(model, tokenizer, model_alias):
    """Exports the model to ONNX format."""
    model.eval().to("cpu")
    sample_input = tokenizer("test", return_tensors="pt")
    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]
    
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, f"{model_alias}.onnx")
    
    torch.onnx.export(
        model, 
        (sample_input["input_ids"], sample_input["attention_mask"]), 
        onnx_path, 
        input_names=input_names, 
        output_names=output_names, 
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"}, 
            "attention_mask": {0: "batch", 1: "sequence"}, 
            "output": {0: "batch"}
        }
    )
    print(f"ONNX model exported to {onnx_path}")
    return onnx_path

# %%
def get_model_size(model, model_path=None):
    """Get the size of the model in MB."""
    if model_path:
        # Get the size of the saved model file
        size_bytes = os.path.getsize(model_path)
    else:
        # Estimate the size in memory
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        size_bytes = param_size
    
    # Convert to MB
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

# %% [markdown]
# ### Post-Training Quantization (PTQ) Functions

# %%
def apply_dynamic_quantization(model, model_alias):
    """Apply dynamic quantization to the model."""
    # Make a copy of the model for quantization
    quantized_model = copy.deepcopy(model)
    quantized_model.eval()
    
    # Move to CPU as quantization is only supported on CPU
    quantized_model.to("cpu")
    
    # Apply dynamic quantization
    quantized_model = torch.ao.quantization.quantize_dynamic(
        quantized_model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save the quantized model
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_alias}.pth")
    torch.save(quantized_model.state_dict(), model_path)
    
    print(f"Dynamic quantized model saved to {model_path}")
    return quantized_model, model_path

# %%
def apply_static_quantization(model, calibration_loader, model_alias):
    """
    Apply static quantization to the model.
    Static quantization requires a calibration step with representative data.
    """
    # Make a copy of the model for quantization
    quantized_model = copy.deepcopy(model)
    # Set the quantization engine based on your hardware
    import platform
    if platform.machine() in ['x86_64', 'AMD64']:
        torch.backends.quantized.engine = 'fbgemm'
    else:
        torch.backends.quantized.engine = 'qnnpack'
    torch.backends.quantized.engine = 'qnnpack'
    
    quantized_model.eval()
    
    # Move to CPU as quantization is only supported on CPU
    quantized_model.to("cpu")
    
    # Define qconfig mapping
    qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_qconfig)
    
    # Prepare model for calibration
    prepared_model = torch.quantization.prepare(quantized_model, qconfig_mapping)
    
    # Calibrate with a subset of data
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            input_ids, attention_mask, _ = batch
            prepared_model(input_ids, attention_mask)
            # Use only a small subset for calibration
            if batch_idx >= 10:
                break
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)
    
    # Save the quantized model
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_alias}.pth")
    torch.save(quantized_model.state_dict(), model_path)
    
    print(f"Static quantized model saved to {model_path}")
    return quantized_model, model_path

# %%
def apply_weight_only_quantization(model, model_alias):
    """Apply weight-only quantization to the model."""
    # Make a copy of the model for quantization
    quantized_model = copy.deepcopy(model)
    quantized_model.eval()
    
    # Move to CPU as quantization is only supported on CPU
    quantized_model.to("cpu")
    
    # Set qconfig for weight-only quantization
    qconfig_mapping = QConfigMapping().set_global(float_qparams_weight_only_qconfig)
    
    # Prepare and convert model
    prepared_model = torch.quantization.prepare(quantized_model, qconfig_mapping)
    quantized_model = torch.quantization.convert(prepared_model)
    
    # Save the quantized model
    model_dir = f"model-metric/{model_alias}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_alias}.pth")
    torch.save(quantized_model.state_dict(), model_path)
    
    print(f"Weight-only quantized model saved to {model_path}")
    return quantized_model, model_path

# %% [markdown]
# ## Quantization-Aware Training (QAT) Functions

# %%
class QuantizationAwareTraining:
    def __init__(self, model, train_loader, test_loader, label_encoder, model_alias, epochs=3, learning_rate=5e-5, class_weights=None):
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.label_encoder = label_encoder
        self.model_alias = model_alias
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.device = torch.device("cpu")  # QAT requires CPU
        
        # Create directory for model metrics
        self.model_dir = f"model-metric/{model_alias}"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.model_dir)
    
    def prepare_qat_model(self):
        """Prepare model for Quantization-Aware Training."""
        # Set model to train mode
        self.model.train()
        self.model.to(self.device)
        
        # Define qconfig mapping for QAT
        qconfig_mapping = QConfigMapping().set_global(torch.quantization.get_default_qat_qconfig("fbgemm"))
        
        # Prepare model for QAT
        self.qat_model = torch.quantization.prepare_qat(self.model, qconfig_mapping)
        return self.qat_model
    
    def train_qat_model(self):
        """Train the model with Quantization-Aware Training."""
        # Set up loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(self.device) if self.class_weights is not None else None)
        optimizer = torch.optim.AdamW(self.qat_model.parameters(), lr=self.learning_rate)
        
        epoch_losses = []
        metrics_data = []
        
        for epoch in range(self.epochs):
            start_time = time.time()
            total_loss = 0
            all_preds, all_labels = [], []
            
            for batch_idx, batch in enumerate(self.train_loader):
                input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                
                optimizer.zero_grad()
                outputs = self.qat_model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().tolist()
                labels = labels.cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                # Log batch loss every 10 batches
                if batch_idx % 10 == 0:
                    self.writer.add_scalar("BatchLoss/qat_train", loss.item(), epoch * len(self.train_loader) + batch_idx)
            
            # Compute epoch metrics
            avg_loss = total_loss / len(self.train_loader)
            epoch_losses.append(avg_loss)
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
            epoch_time = time.time() - start_time
            
            # Store metrics for CSV logging
            metrics_data.append([epoch + 1, avg_loss, accuracy, precision, recall, f1, epoch_time])
            
            # Print metrics
            print(f"QAT Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}, Time={epoch_time:.2f}s")
            
            # Log metrics to TensorBoard
            self.writer.add_scalar("Loss/qat_train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/qat_train", accuracy, epoch)
            self.writer.add_scalar("Precision/qat_train", precision, epoch)
            self.writer.add_scalar("Recall/qat_train", recall, epoch)
            self.writer.add_scalar("F1-score/qat_train", f1, epoch)
            self.writer.add_scalar("Time/Epoch_qat", epoch_time, epoch)
        
        # Save model KPIs as CSV
        metrics_df = pd.DataFrame(metrics_data, columns=["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"])
        metrics_df.to_csv(os.path.join(self.model_dir, "qat_training_metrics.csv"), index=False)
        
        # Save training loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
        plt.title('QAT Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        loss_plot_path = os.path.join(self.model_dir, "qat_training_loss.png")
        plt.savefig(loss_plot_path)
        self.writer.add_figure("QAT Training Loss", plt.gcf(), close=True)
        
        # Convert model to quantized model
        self.quantized_model = torch.quantization.convert(self.qat_model.eval())
        
        # Save model weights
        model_path = os.path.join(self.model_dir, f"{self.model_alias}.pth")
        torch.save(self.quantized_model.state_dict(), model_path)
        
        self.writer.flush()
        self.writer.close()
        
        return self.quantized_model, model_path

# %% [markdown]
# ### Comparison Utilities

# %%
def measure_inference_latency(model, test_loader, num_batches=100):
    """Measure inference latency for a model."""
    model.eval()
    device = torch.device("cpu")  # Ensure we're on CPU for fair comparison
    model.to(device)
    
    latencies = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
                
            input_ids, attention_mask, _ = [x.to(device) for x in batch]
            
            # Warm-up run
            _ = model(input_ids, attention_mask=attention_mask)
            
            # Timed run
            start_time = time.time()
            _ = model(input_ids, attention_mask=attention_mask)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
    
    # Calculate average latency in milliseconds
    avg_latency_ms = np.mean(latencies) * 1000
    return avg_latency_ms

def compare_models(models_info):
    """Compare multiple models and create visualizations."""
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame(models_info)
    
    # Save comparison to CSV
    comparison_df.to_csv("model_comparison.csv", index=False)
    
    # Create bar charts for visual comparison
    metrics = ['Size (MB)', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Latency (ms)']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(comparison_df['Model'], comparison_df[metric])
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} Across Models')
        plt.xticks(rotation=45)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"comparison_{metric.replace(' ', '_').lower()}.png")
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df



# %% [markdown]
# ## Main Execution

# %% [markdown]
# ### Step 1: Setup

# %%

MODEL_NAME = "distilbert/distilbert-base-uncased"

# Define model aliases
original_model_alias = 'distilbert-original'
dynamic_ptq_model_alias = 'distilbert-dynamic-ptq'
static_ptq_model_alias = 'distilbert-static-ptq'
weight_only_ptq_model_alias = 'distilbert-weight-only-ptq'
qat_model_alias = 'distilbert-qat'

update_model_dict(original_model_alias, MODEL_NAME)
update_model_dict(dynamic_ptq_model_alias, MODEL_NAME)
update_model_dict(static_ptq_model_alias, MODEL_NAME)
update_model_dict(weight_only_ptq_model_alias, MODEL_NAME)
update_model_dict(qat_model_alias, MODEL_NAME)

# %% [markdown]
# ### Step 2: Data Loading and Preprocessing

# %%

df, label_encoder = load_and_preprocess_data(model_alias=original_model_alias)
balanced_df = balance_dataset(df)
balanced_df['conversation'] = balanced_df['conversation'].apply(preprocess_conversation)

# %% [markdown]
# ### Step 3: Create DataLoaders

# %%

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
train_loader, test_loader, test_df = create_dataloaders(balanced_df, tokenizer)

# %% [markdown]
# ### Step 4: Initialize Model

# %%
num_classes = len(label_encoder.classes_)
model = DistilBERTWithLoRA(num_labels=num_classes)
class_weights = compute_class_weights(balanced_df['labels'], num_classes)  

# %% [markdown]
# ### Step 5: Train and Evaluate Original Model

# %%
print("\n=== Training Original Model ===")
train_model(model, train_loader, model_alias=original_model_alias, epochs=5, learning_rate=5e-5, class_weights=class_weights)

# %%
# Save original model
original_model_path = f"model-metric/{original_model_alias}/{original_model_alias}.pth"

# %%
# Evaluate original model
print("\n=== Evaluating Original Model ===")
original_precision, original_recall, original_f1, original_eval_time, _, _ = evaluate_model(
    model, test_loader, label_encoder, original_model_alias
)

# %%
# Measure model size and latency
original_size = get_model_size(model, original_model_path)

original_latency = measure_inference_latency(model, test_loader)
print(f"original_size: {original_size}")
print(f"original_latency: {original_latency}")

# %%
# Get accuracy from test set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to("cpu") for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        labels = labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
original_accuracy = accuracy_score(all_labels, all_preds)

# %% [markdown]
# ### Step 6: Apply Dynamic Post-Training Quantization

# %%
print("\n=== Applying Dynamic Post-Training Quantization ===")
dynamic_quantized_model, dynamic_quantized_path = apply_dynamic_quantization(model, dynamic_ptq_model_alias)

# %%
# Evaluate dynamic quantized model
print("\n=== Evaluating Dynamic Quantized Model ===")
dynamic_precision, dynamic_recall, dynamic_f1, dynamic_eval_time, _, _ = evaluate_model(
    dynamic_quantized_model, test_loader, label_encoder, dynamic_ptq_model_alias
)
    

# %%
# Measure model size and latency
dynamic_size = get_model_size(dynamic_quantized_model, dynamic_quantized_path)
dynamic_latency = measure_inference_latency(dynamic_quantized_model, test_loader)

# %%
# Get accuracy from test set
dynamic_quantized_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to("cpu") for x in batch]
        outputs = dynamic_quantized_model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        labels = labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
dynamic_accuracy = accuracy_score(all_labels, all_preds)

# %% [markdown]
# ### Step 7: Apply Static Post-Training Quantization

# %%

print("\n=== Applying Static Post-Training Quantization ===")
# Use a subset of train_loader for calibration
calibration_loader = DataLoader(
    CustomDataset(balanced_df.sample(100), tokenizer),
    batch_size=8,
    shuffle=True
)
static_quantized_model, static_quantized_path = apply_static_quantization(model, calibration_loader, static_ptq_model_alias)


# %%
# Evaluate static quantized model
print("\n=== Evaluating Static Quantized Model ===")
static_precision, static_recall, static_f1, static_eval_time, _, _ = evaluate_model(
    static_quantized_model, test_loader, label_encoder, static_ptq_model_alias
)

# %%
# Measure model size and latency
static_size = get_model_size(static_quantized_model, static_quantized_path)
static_latency = measure_inference_latency(static_quantized_model, test_loader)

# %%
print(f"static_size: {static_size}")
print(f"static_latency: {static_latency}")

# %%
 # Get accuracy from test set
static_quantized_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to("cpu") for x in batch]
        outputs = static_quantized_model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        labels = labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
static_accuracy = accuracy_score(all_labels, all_preds)

# %%
print(f"static_accuracy: {static_accuracy}")

# %% [markdown]
# ### Step 8: Apply Weight-Only Quantization

# %%
print("\n=== Applying Weight-Only Quantization ===")
weight_only_model, weight_only_path = apply_weight_only_quantization(model, weight_only_ptq_model_alias)

# Evaluate weight-only quantized model
print("\n=== Evaluating Weight-Only Quantized Model ===")
weight_only_precision, weight_only_recall, weight_only_f1, weight_only_eval_time, _, _ = evaluate_model(
    weight_only_model, test_loader, label_encoder, weight_only_ptq_model_alias
)

# %%
# Measure model size and latency
weight_only_size = get_model_size(weight_only_model, weight_only_path)
weight_only_latency = measure_inference_latency(weight_only_model, test_loader)

# %%
print(f"weight_only_size: {weight_only_size}")
print(f"weight_only_latency: {weight_only_latency}")

# %%
# Get accuracy from test set
weight_only_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to("cpu") for x in batch]
        outputs = weight_only_model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        labels = labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
weight_only_accuracy = accuracy_score(all_labels, all_preds)

# %%
print(f"weight_only_accuracy: {weight_only_accuracy}")

# %% [markdown]
#  ### Step 9: Apply Quantization-Aware Training

# %%

print("\n=== Applying Quantization-Aware Training ===")
qat = QuantizationAwareTraining(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    label_encoder=label_encoder,
    model_alias=qat_model_alias,
    epochs=5,
    learning_rate=5e-5,
    class_weights=class_weights
)
qat_model = qat.prepare_qat_model()
qat_quantized_model, qat_path = qat.train_qat_model()

# Evaluate QAT model
print("\n=== Evaluating QAT Model ===")
qat_precision, qat_recall, qat_f1, qat_eval_time, _, _ = evaluate_model(
    qat_quantized_model, test_loader, label_encoder, qat_model_alias
)

# Measure model size and latency
qat_size = get_model_size(qat_quantized_model, qat_path)
qat_latency = measure_inference_latency(qat_quantized_model, test_loader)

# Get accuracy from test set
qat_quantized_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to("cpu") for x in batch]
        outputs = qat_quantized_model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().tolist()
        labels = labels.cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels)
qat_accuracy = accuracy_score(all_labels, all_preds)

# %% [markdown]
# ### Step 10: Compare All Models

# %%

models_info = [
    {
        'Model': 'Original',
        'Size (MB)': original_size,
        'Accuracy': original_accuracy,
        'Precision': original_precision,
        'Recall': original_recall,
        'F1-Score': original_f1,
        'Latency (ms)': original_latency
    },
    {
        'Model': 'Dynamic PTQ',
        'Size (MB)': dynamic_size,
        'Accuracy': dynamic_accuracy,
        'Precision': dynamic_precision,
        'Recall': dynamic_recall,
        'F1-Score': dynamic_f1,
        'Latency (ms)': dynamic_latency
    },
    {
        'Model': 'Static PTQ',
        'Size (MB)': static_size,
        'Accuracy': static_accuracy,
        'Precision': static_precision,
        'Recall': static_recall,
        'F1-Score': static_f1,
        'Latency (ms)': static_latency
    },
    {
        'Model': 'Weight-Only PTQ',
        'Size (MB)': weight_only_size,
        'Accuracy': weight_only_accuracy,
        'Precision': weight_only_precision,
        'Recall': weight_only_recall,
        'F1-Score': weight_only_f1,
        'Latency (ms)': weight_only_latency
    },
    {
        'Model': 'QAT',
        'Size (MB)': qat_size,
        'Accuracy': qat_accuracy,
        'Precision': qat_precision,
        'Recall': qat_recall,
        'F1-Score': qat_f1,
        'Latency (ms)': qat_latency
    }
]

comparison_df = compare_models(models_info)

# %% [markdown]
# ### Save tokenizer for all models

# %%

for model_alias in [original_model_alias, dynamic_ptq_model_alias, static_ptq_model_alias, weight_only_ptq_model_alias, qat_model_alias]:
    tokenizer.save_pretrained(f"model-metric/{model_alias}/tokenizer/")


