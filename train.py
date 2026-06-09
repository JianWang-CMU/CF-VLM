import os
import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info

# 数据预处理函数（保持原样）
def process_func(example):
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"{file_path}", "resized_height": 280, "resized_width": 280},
            {"type": "text", "text": "COCO Yes:"},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs
    response = tokenizer(output_content, add_special_tokens=False)
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs['pixel_values']),
        "image_grid_thw": torch.tensor(inputs['image_grid_thw']).squeeze(0),
    }

if __name__ == "__main__":
    model_path = "Qwen2.5-VL-Finetune/Qwen/Qwen2___5-VL-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    peft_model = get_peft_model(model, config)

    # 加载原始大数据
    with open("Qwen2.5-VL-Finetune/flickr30k/train_flickr_6000.json", "r") as f:
        all_data = json.load(f)

    split_size = 500
    for i in range(0, len(all_data), split_size):
        part_id = i // split_size + 1
        print(f"\n🚀 正在训练第 {part_id} 部分（样本 {i} 到 {i + split_size - 1}）...")

        split_data = all_data[i:i + split_size]
        dataset = Dataset.from_list(split_data).map(process_func)

        args = TrainingArguments(
            output_dir=f"./output/Qwen2.5-VL-7B/part_{part_id}",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=10,
            num_train_epochs=1,
            save_steps=100,
            learning_rate=1e-4,
            save_on_each_node=True,
            gradient_checkpointing=True,
            report_to="none",
        )

        trainer = Trainer(
            model=peft_model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )

        trainer.train()

        # 保存当前批次训练后的模型
        save_dir = f"./output/Qwen2.5-VL-7B/final_model_part_{part_id}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"💾 正在保存第 {part_id} 部分模型到 {save_dir} ...")
        peft_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)

    print("✅ 所有批次训练完成！")

    # exit()

    # # ====================测试模式===================
    # # 配置测试参数
    # val_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     inference_mode=True,  # 测试模式
    #     r=64,  # Lora 秩
    #     lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    #     lora_dropout=0.05,  # Dropout 比例
    #     bias="none",
    # )

    # # 获取测试模型
    # val_peft_model = PeftModel.from_pretrained(model, model_id="./output/Qwen2.5-VL-7B/checkpoint-56", config=val_config)

    # # 读取测试数据
    # with open("coco_2014/data_vl_test.json", "r") as f:
    #     test_dataset = json.load(f)

    # test_image_list = []
    # for item in test_dataset:
    #     input_image_prompt = item["conversations"][0]["value"]
    #     # 去掉前后的<|vision_start|>和<|vision_end|>
    #     # origin_image_path = input_image_prompt.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    #     origin_image_path = input_image_prompt      
    #     messages = [{
    #         "role": "user", 
    #         "content": [
    #             {
    #             "type": "image", 
    #             "image": origin_image_path
    #             },
    #             {
    #             "type": "text",
    #             "text": "COCO Yes:"
    #             }
    #         ]}]
        
    #     response = predict(messages, val_peft_model)
    #     messages.append({"role": "assistant", "content": f"{response}"})
    #     print(messages[-1])

    #     # test_image_list.append(swanlab.Image(origin_image_path, caption=response))

    # # swanlab.log({"Prediction": test_image_list})

    # # # 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
    # # swanlab.finish()
