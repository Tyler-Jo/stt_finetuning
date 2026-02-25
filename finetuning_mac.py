import os
import ssl
import requests
import torch
import librosa
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from urllib3.exceptions import InsecureRequestWarning

# 1. SSL 및 보안 설정
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = "./train_data"
MODEL_PATH = "./whisper-tiny-local"
OUTPUT_DIR = "./whisper-tiny-finetuned"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def prepare_dataset_batched(batch, processor):
    audio_paths = [os.path.join(BASE_DIR, "audio", f) for f in batch["file_name"]]
    speech_list = [librosa.load(p, sr=16000)[0] for p in audio_paths]
    inputs = processor.feature_extractor(speech_list, sampling_rate=16000)
    batch["input_features"] = inputs.input_features
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "metadata.csv"))
    dataset = Dataset.from_pandas(df)

    # 6. 모델 및 프로세서 로드
    print("🤖 순정 모델 및 프로세서 로드 중...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)


    dataset = dataset.map(
        prepare_dataset_batched,
        batched=True,
        batch_size=16,
        fn_kwargs={"processor": processor},
        remove_columns=dataset.column_names,
        keep_in_memory=False, 
    )
    dataset = dataset.train_test_split(test_size=0.1)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=1,
        max_steps=500,
        gradient_checkpointing=True,
        fp16=False,                  # MPS 환경 고려
        predict_with_generate=True,
        generation_max_length=225,
        eval_strategy="steps",       
        save_strategy="steps",       
        eval_steps=100,              # 100 step마다 평가
        save_steps=100,              # 100 step마다 저장
        logging_steps=10,
        save_total_limit=3,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    )

    print("학습 시작")
    trainer.train()

    print("최종 모델 가중치 저장 중...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print(f"학습 완료 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()