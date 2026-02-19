import torch
import numpy as np
import sounddevice as sd
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    pipeline,
    GenerationConfig,
)

# 1. 경로 설정
MODEL_PATH = "./whisper-tiny-finetuned"

# 2. 모델 및 프로세서 개별 로드 (우리가 찾아낸 필승 조합)
print("🚀 모델 로딩 중... 잠시만 기다려 주세요.")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 한국어 설정 강제 주입
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="korean", task="transcribe"
)
model.generation_config = GenerationConfig.from_model_config(model.config)
model.generation_config.update(language="korean", task="transcribe")

# 3. 파이프라인 구축
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    device=device,
)

# 4. 실시간 녹음 및 추론 설정
SAMPLING_RATE = 16000
DURATION = 5  # 5초 단위로 끊어서 인식


def record_and_transcribe():
    print(f"\n🎤 {DURATION}초 동안 말씀해 주세요... (종료하려면 Ctrl+C)")

    while True:
        try:
            # 마이크로부터 데이터 수집
            recording = sd.rec(
                int(DURATION * SAMPLING_RATE),
                samplerate=SAMPLING_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()  # 녹음이 끝날 때까지 대기

            # 2차원 배열을 1차원으로 변환
            audio_data = recording.flatten()

            # 추론
            result = pipe({"raw": audio_data, "sampling_rate": SAMPLING_RATE})

            print(f"📝 인식 결과: {result['text']}")
            print("---")

        except KeyboardInterrupt:
            print("\n👋 실시간 테스트를 종료합니다.")
            break


if __name__ == "__main__":
    record_and_transcribe()
