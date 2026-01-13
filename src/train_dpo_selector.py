import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import os

class PreferenceDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer, device):
    queries = [item['query'] for item in batch]
    chosens = [item['chosen'] for item in batch]
    rejecteds = [item['rejected'] for item in batch]

    chosen_inputs = tokenizer(
        queries, chosens,
        padding=True, truncation=True, max_length=512, return_tensors='pt'
    ).to(device)

    rejected_inputs = tokenizer(
        queries, rejecteds,
        padding=True, truncation=True, max_length=512, return_tensors='pt'
    ).to(device)

    return chosen_inputs, rejected_inputs

def evaluate(model, dataloader, device):
    """Calculates accuracy (how often Chosen Score > Rejected Score)"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for chosen_inputs, rejected_inputs in dataloader:
            with torch.cuda.amp.autocast():
                chosen_scores = model(**chosen_inputs).logits
                rejected_scores = model(**rejected_inputs).logits
            
            correct += (chosen_scores > rejected_scores).sum().item()
            total += chosen_scores.size(0)
            
    return correct / total

from torch.utils.tensorboard import SummaryWriter

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    writer = SummaryWriter(log_dir=args.log_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)

    full_dataset = PreferenceDataset(args.dataset_path)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer, device)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=lambda b: collate_fn(b, tokenizer, device)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    print("Starting training...")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for chosen_inputs, rejected_inputs in progress_bar:
            optimizer.zero_grad()
            global_step += 1

            with torch.cuda.amp.autocast():
                chosen_scores = model(**chosen_inputs).logits
                rejected_scores = model(**rejected_inputs).logits
                
                diff = chosen_scores - rejected_scores
                loss = -torch.nn.functional.logsigmoid(diff).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            writer.add_scalar("Train/StepLoss", loss.item(), global_step)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        val_acc = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        writer.add_scalar("Train/EpochLoss", avg_loss, epoch + 1)
        writer.add_scalar("Val/Accuracy", val_acc, epoch + 1)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  - Avg Loss: {avg_loss:.4f}")
        print(f"  - Validation Accuracy (Preference): {val_acc:.2%}") 

    writer.close()
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dpo_selector_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--log_dir", type=str, default="runs/dpo_experiment")

    args = parser.parse_args()
    
    if os.path.exists(args.dataset_path):
        train(args)
    else:
        print(f"Error: Dataset {args.dataset_path} not found.")