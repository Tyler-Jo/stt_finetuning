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
    GenerationConfig,
)
from urllib3.exceptions import InsecureRequestWarning

# 1. SSL 및 보안 설정 (보안망 환경 대응)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# 2. 경로 설정
BASE_DIR = "./train_data"
MODEL_PATH = "./whisper-tiny-local"
OUTPUT_DIR = "./whisper-tiny-finetuned"


# 3. Whisper 전용 데이터 콜레이터 정의
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 입력 오디오 특징(input_features) 패딩
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 라벨(labels) 패딩
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 손실 계산 시 패딩 토큰 무시 (-100)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


# 4. 메모리 효율적 전처리 함수 (Batched)
def prepare_dataset_batched(batch, processor):
    # 오디오 경로 생성 및 로드
    audio_paths = [os.path.join(BASE_DIR, "audio", f) for f in batch["file_name"]]
    # librosa로 16kHz 로드 (직접 디코딩하여 torchcodec 에러 방지)
    speech_list = [librosa.load(p, sr=16000)[0] for p in audio_paths]

    # 특징 벡터 추출
    inputs = processor.feature_extractor(speech_list, sampling_rate=16000)
    batch["input_features"] = inputs.input_features

    # 텍스트 토큰화 (metadata.csv의 'text' 컬럼 사용)
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


def main():
    # 5. 데이터셋 로드
    print("📦 데이터를 불러오는 중...")
    if not os.path.exists(os.path.join(BASE_DIR, "metadata.csv")):
        print("❌ metadata.csv 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(os.path.join(BASE_DIR, "metadata.csv"))  #
    dataset = Dataset.from_pandas(df)

    # 6. 모델 및 프로세서 로드
    print("🤖 로컬 모델 로드 중...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH, local_files_only=True
    )

    # 7. 전처리 적용 (RAM 절약 모드)
    print("🧹 데이터 전처리 시작 (Batched)...")
    dataset = dataset.map(
        prepare_dataset_batched,
        batched=True,
        batch_size=16,  # 메모리 상황에 따라 조절
        fn_kwargs={"processor": processor},
        remove_columns=dataset.column_names,
        num_proc=1,
        keep_in_memory=False,  # 디스크 캐시 활용
    )

    # 학습/검증 데이터 분리 (9:1)
    dataset = dataset.train_test_split(test_size=0.1)

    # 8. 콜레이터 및 학습 설정
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # VRAM 부족 시 2로 낮춤
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=1,
        max_steps=10,  # 전체 학습 루프 횟수
        gradient_checkpointing=True,
        fp16=True,  # GPU 사용 시 True
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=1,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
    )

    # 9. 트레이너 실행
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        # tokenizer=processor.feature_extractor,
    )

    print("🚀 학습 시작...")
    trainer.train()

    # 3. 🔥 핵심: 모든 설정 파일을 완벽하게 저장
    print("💾 모델 및 설정 파일 저장 중...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # 최신 규격의 Generation Config를 생성해서 함께 저장
    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.update(language="korean", task="transcribe")
    gen_config.save_pretrained(OUTPUT_DIR)

    print(f"✅ 모든 준비 완료! 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
