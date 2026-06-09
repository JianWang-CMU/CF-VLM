import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm

# === 加载 flickr30k 格式训练集 ===
class FlickrImageTextDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["conversations"][0]["value"]
        text = item["conversations"][1]["value"]
        image = Image.open(image_path).convert("RGB")
        processed = self.processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
        return processed["pixel_values"].squeeze(0), processed["input_ids"].squeeze(0)

def collate_fn(batch):
    images, input_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    max_len = max(x.size(0) for x in input_ids)
    padded_ids = torch.stack([F.pad(x, (0, max_len - x.size(0)), value=0) for x in input_ids], dim=0)
    return images, padded_ids

class SimpleCLIPTrainer(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)

    def forward(self, images, input_ids):
        device = next(self.parameters()).device
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long().to(device)
        inputs = {
            "pixel_values": images.to(device),
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask,
        }
        outputs = self.model(**inputs)
        img_embeds = F.normalize(outputs.image_embeds, dim=-1)
        txt_embeds = F.normalize(outputs.text_embeds, dim=-1)
        logits_per_image = img_embeds @ txt_embeds.T
        labels = torch.arange(len(images), device=device)
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_image.T, labels)) / 2
        return loss

@torch.no_grad()
def predict_image_text_similarity(model, processor, image_path, text, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    image_embeds = F.normalize(outputs.image_embeds, dim=-1)
    text_embeds = F.normalize(outputs.text_embeds, dim=-1)
    return torch.sum(image_embeds * text_embeds).item()

# === 保持原 ConMe 格式评估逻辑 ===
def evaluate_on_subset(model, processor, device, test_json_path, subset_size=3000):
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_data = test_data[:subset_size]

    model.eval()
    correct = 0
    for item in test_data:
        image_path = item["conversations"][0]["value"]
        question_full = item["conversations"][1]["value"]
        gt_answer = item["conversations"][2]["value"].strip().upper()

        try:
            question_main, options = question_full.split("A:", 1)
            option_a, option_b = options.split("B:")
            prompt_a = question_main.strip() + " A: " + option_a.strip()
            prompt_b = question_main.strip() + " B: " + option_b.strip()
        except:
            prompt_a = question_full.strip() + " A"
            prompt_b = question_full.strip() + " B"

        sim_a = predict_image_text_similarity(model, processor, image_path, prompt_a, device)
        sim_b = predict_image_text_similarity(model, processor, image_path, prompt_b, device)

        pred = "A" if sim_a > sim_b else "B"
        if pred == gt_answer:
            correct += 1

    acc = correct / subset_size
    print(f"\n🧪 [ConMe评估] 前 {subset_size} 样本准确率: {acc:.2%}\n")
    model.train()

def train(model, dataloader, optimizer, device, eval_path, num_epochs=1000, eval_interval=3000):
    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, input_ids in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            loss = model(images, input_ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            step += len(images)
            if step >= eval_interval:
                print(f"\n📉 累计训练 {step} 样本，触发 ConMe 评估：")
                evaluate_on_subset(model.model, model.processor, device, eval_path)
                step = 0  # reset step

        print(f"✅ Epoch {epoch+1} 平均损失: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    train_data_path = "Qwen2.5-VL-Finetune/flickr30k/flickr_captions_1500_first_images.json"
    eval_data_path = "Qwen2.5-VL-Finetune/ConMe_data/att.json"  # 不变

    with open(train_data_path, "r") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCLIPTrainer().to(device)
    dataset = FlickrImageTextDataset(data, model.processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)
    train(model, dataloader, optimizer, device, eval_path=eval_data_path, num_epochs=1000, eval_interval=3000)

    save_path = "./clip_finetuned_flickr"
    os.makedirs(save_path, exist_ok=True)
    model.model.save_pretrained(save_path)
    model.processor.save_pretrained(save_path)
    print(f"✅ 模型已保存至: {save_path}")

