import os, json, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datasets import Dataset
from diffusers import StableDiffusionPipeline,DiffusionPipeline
from transformers import (
    AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

# ------------------ 配置 ------------------
model_path = "/Qwen2.5-VL-Finetune/Qwen/Qwen2___5-VL-7B-Instruct"
train_json_path = "coco_2014/data_vl_train.json"
cf_dir = "cf_data"
os.makedirs(cf_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
processor = AutoProcessor.from_pretrained(model_path)

from openai import OpenAI

# 请替换成您的 DeepSeek API 密钥
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

# 生成反事实文本的函数
def generate_text_with_deepseekv3(prompt):
    try:
        # 通过 DeepSeek API 调用模型生成反事实文本
        response = client.chat.completions.create(
            extra_body={},
            model="qwen/qwen-turbo",  # DeepSeek 使用的模型名称
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Your task is to generate ONE detailed counterfactual text that meaningfully inverts key elements of the input text. The counterfactual should:1. **Invert multiple aspects of the original scenario**: This includes actions, events, participants, outcomes, context, and underlying assumptions.2. **Provide a plausible alternative**: Create a detailed alternative reality where key variables or circumstances differ significantly from the original. This includes different actors, places, or outcomes that logically follow from the changes in the scenario.3. **Ensure natural progression and coherence**: The counterfactual text should not only provide an inverted scenario but should also include sufficient background and explanation for why and how the changes occur. The new narrative should be consistent and flow logically, given the changes.4. **Present contrasting details**: Highlight the changes between the original and the counterfactual reality in terms of behavior, motivation, environment, or other relevant factors. These changes should be not just superficial but meaningful, affecting the core dynamics of the situation.5. **Include rich details**: Be specific and descriptive about the settings, emotional responses, consequences, and interactions in the counterfactual scenario.Your response should only be the generated counterfactual text. Do not include any additional commentary or explanations, just provide the rewritten version of the scenario with all the requested changes in place."},
                  {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        # 获取生成的文本并返回
        generated_text = response.choices[0].message.content  # 修改为正确的访问方式
        return generated_text
    
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return ""


def load_pipeline(cls, model_id, dtype, device="cuda"):
    try:
        # 优先尝试从本地加载
        pipe = cls.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True).to(device)
        print(f"Loaded {model_id} from local cache.")
    except Exception as e:
        print(f"Local load failed for {model_id}, downloading... ({e})")
        pipe = cls.from_pretrained(model_id, torch_dtype=dtype).to(device)
    return pipe

# ------------------ 第一步：生成反事实样本 ------------------
def gen_cf_data(K=5):
    print("🧠 正在加载训练数据集 ...")
    dataset = Dataset.from_json(train_json_path)
    
    # 加载Stable Diffusion模型用于生成反事实图像
    print("🖼️ 正在加载 Stable Diffusion 模型 ...")
    sd_pipe = load_pipeline(DiffusionPipeline, "stabilityai/stable-diffusion-xl-base-1.0", torch.float16)
    
    output_data = []
    print(f"💬 开始生成 {K} 个反事实样本 ...")

    for idx, item in enumerate(dataset):
        conv = item["conversations"]
        orig_text = conv[0]["value"]
        orig_output = conv[1]["value"]
        cf_samples = []

        for k in range(K):
            # 只扰动输出，不对prompt扰动
            prompt = f"将下面这句话改写为语义相反或细微差别的表述：\n\"{orig_output}\""  # 对输出进行扰动
            cf_text = generate_text_with_deepseekv3(prompt)  # 通过DeepSeekV3生成反事实文本

            # 生成反事实图像
            cf_img = sd_pipe(cf_text, guidance_scale=7.5).images[0]
            cf_img_path = os.path.join(cf_dir, f"cf_img_{idx}_{k}.png")
            cf_img.save(cf_img_path)
            cf_samples.append({"cf_text": cf_text, "cf_image_path": cf_img_path})

        output_data.append({
            "orig_input": orig_text,
            "orig_output": orig_output,
            "cf_samples": cf_samples
        })

    # 释放Stable Diffusion模型
    print("🧹 正在释放 Stable Diffusion 模型 ...")
    del sd_pipe
    torch.cuda.empty_cache()

    # 保存反事实数据
    print("💾 正在保存反事实数据 ...")
    with open(os.path.join(cf_dir, "cf_processed.json"), "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

# ------------------ 第二步：预处理函数 ------------------
# ------------------ 第二步：预处理函数 ------------------
def process_func(example):
    def extract_image_path(text):
        # 提取图像路径
        return text.split("<|vision_start|>")[1].split("<|vision_end|>")[0].strip()

    def encode(image_path, prompt_text, label_text):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path), "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": "COCO Yes:"}
            ]
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_input, vid_input = process_vision_info(messages)
        proc = processor(text=[prompt], images=img_input, videos=vid_input, return_tensors="pt", padding=True)
        label_tok = tokenizer(label_text.strip(), add_special_tokens=False)
        iid = proc.input_ids[0].tolist() + label_tok.input_ids + [tokenizer.pad_token_id]
        att = proc.attention_mask[0].tolist() + label_tok.attention_mask + [1]
        lab = [-100] * len(proc.input_ids[0]) + label_tok.input_ids + [tokenizer.pad_token_id]
        return iid[:8192], att[:8192], lab[:8192], proc.pixel_values, proc.image_grid_thw.squeeze(0)

    # 正样本图文对
    orig_input = example["orig_input"]
    orig_output = example["orig_output"]
    image_path = orig_input.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    iid, att, lab, pix, grid = encode(image_path, orig_input, orig_output)

    # 反事实样本图文对
    cf_iids, cf_atts, cf_labs, cf_pixs, cf_grids = [], [], [], [], []
    cf_texts = [cf["cf_text"] for cf in example["cf_samples"]]
    cf_imgs = [cf["cf_image_path"] for cf in example["cf_samples"]]
    
    # 修改：直接使用 orig_output 作为 cf_ans
    cf_ans = orig_output

    for cf_text, cf_img_path in zip(cf_texts, cf_imgs):
        iid_cf, att_cf, lab_cf, pix_cf, grid_cf = encode(cf_img_path, cf_text, cf_ans)
        cf_iids.append(torch.tensor(iid_cf))
        cf_atts.append(torch.tensor(att_cf))
        cf_labs.append(torch.tensor(lab_cf))
        cf_pixs.append(pix_cf)
        cf_grids.append(grid_cf)

    return {
        "input_ids": torch.tensor(iid), "attention_mask": torch.tensor(att), "labels": torch.tensor(lab),
        "pixel_values": pix, "image_grid_thw": grid,
        "cf_input_ids_list": cf_iids,
        "cf_attention_mask_list": cf_atts,
        "cf_labels_list": cf_labs,
        "cf_pixel_values_list": cf_pixs,
        "cf_image_grid_thw_list": cf_grids
    }


# ------------------ 第三步：加载Qwen模型 ------------------
def load_qwen_model():
    print("🧠 正在加载 Qwen 基础模型中 ...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    base_model.enable_input_require_grads()
    return base_model

# ------------------ 模型结构 ------------------
class CausalVLModel(nn.Module):
    def __init__(self, base_model, lora_rank=16, alpha=1.0, beta=1.0, gamma=0.5):
        super().__init__()
        cfg = LoraConfig(task_type="CAUSAL_LM", r=lora_rank, lora_alpha=32,
                         target_modules=["q_proj", "k_proj", "v_proj"], lora_dropout=0.1, bias="none")
        self.lora = get_peft_model(base_model, cfg)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels,
                cf_input_ids_list, cf_attention_mask_list, cf_pixel_values_list, cf_image_grid_thw_list, cf_labels_list,
                intervene=None):
        B = input_ids.size(0)
        K = len(cf_input_ids_list[0])

        # --- 正例前向 ---
        out = self.lora(input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels)

        # --- 反例前向 ---
        all_cf_feats_img, all_cf_feats_txt = [], []
        for k in range(K):
            input_ids_cf = torch.stack([cf_input_ids_list[i][k] for i in range(B)]).to(input_ids.device)
            att_cf = torch.stack([cf_attention_mask_list[i][k] for i in range(B)]).to(input_ids.device)
            pix_cf = torch.stack([cf_pixel_values_list[i][k] for i in range(B)]).to(input_ids.device)
            grid_cf = torch.stack([cf_image_grid_thw_list[i][k] for i in range(B)]).to(input_ids.device)
            lab_cf = torch.stack([cf_labels_list[i][k] for i in range(B)]).to(input_ids.device)

            out_cf = self.lora(input_ids=input_ids_cf, attention_mask=att_cf,
                            pixel_values=pix_cf, image_grid_thw=grid_cf, labels=lab_cf)

            cf_feat_img = F.normalize(out_cf.logits.mean(dim=1), dim=-1)  # 图像特征（假设mean pool）
            cf_feat_txt = F.normalize(out.logits.mean(dim=1), dim=-1)    # 使用正例文本表示
            all_cf_feats_img.append(cf_feat_img)
            all_cf_feats_txt.append(cf_feat_txt)

        # 原始正对 InfoNCE
        feat_img = F.normalize(out.logits.mean(dim=1), dim=-1)
        feat_txt = F.normalize(out.logits.mean(dim=1), dim=-1)
        logits_i2t = (feat_img @ feat_txt.T) / 0.07
        logits_t2i = (feat_txt @ feat_img.T) / 0.07
        labels = torch.arange(B).to(input_ids.device)
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        loss_contrastive = 0.5 * (loss_i2t + loss_t2i)

        # 对称反事实对比
        loss_negcl = 0.0
        for i in range(B):
            for k in range(K):
                sim_matrix = (all_cf_feats_img[k][i:i+1] @ all_cf_feats_txt[k].T) / 0.07
                loss_1 = F.cross_entropy(sim_matrix, torch.tensor([i], device=input_ids.device))
                sim_matrix_T = (all_cf_feats_img[k].T @ all_cf_feats_txt[k][i:i+1].T).T / 0.07
                loss_2 = F.cross_entropy(sim_matrix_T, torch.tensor([i], device=input_ids.device))
                loss_negcl += 0.5 * (loss_1 + loss_2)
        loss_negcl = loss_negcl / (B * K)

        # 总损失
        loss_total = self.alpha * loss_contrastive + self.beta * loss_negcl

        return {"loss": loss_total}

    def gradient_checkpointing_enable(self, **kwargs):
        self.lora.gradient_checkpointing_enable(**kwargs)

    @property
    def config(self):
        return self.lora.config

    def save_pretrained(self, save_directory, **kwargs):
        self.lora.save_pretrained(save_directory, **kwargs)

# ------------------ 训练流程 ------------------
if __name__ == "__main__":
    print("🚀 开始生成反事实数据 ...")
    gen_cf_data(K=5)  # 第一步：生成反事实数据
    with open(os.path.join(cf_dir, "cf_processed.json"), "r") as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data).map(process_func)

    print("🧠 正在加载并微调Qwen模型 ...")
    # 第二步：加载Qwen模型进行微调
    base_model = load_qwen_model()
    model = CausalVLModel(base_model)
    model.lora.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir="./output/causal_vl", per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=1e-4, save_steps=100, logging_steps=10,
        report_to="none", gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print("🎉 开始训练模型 ...")
    trainer.train()
    print("💾 正在保存微调后的模型 ...")
    save_dir = "./output/causal_vl/final_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"✅ 模型保存成功，路径：{save_dir}")