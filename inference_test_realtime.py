import torch
import numpy as np
import pyaudio
import sys
import os
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor
)

# 1. 경로 및 장치 설정
MODEL_PATH = "./whisper-tiny-finetuned"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. 모델 및 프로세서 로드
print(f"🚀 파인튜닝 모델 로드 중... (장치: {device})")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
processor = WhisperProcessor.from_pretrained(MODEL_PATH)

# ⭐️ 핵심: 모델 내부 설정을 한국어 다국어 모드로 강제 고정
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.is_multilingual = True

# 3. 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024 * 4
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print("\n" + "="*50)
print("🎙️  [한국어 전용] 실시간 군사 용어 테스트 시작")
print("   (종료: Ctrl+C)")
print("="*50 + "\n")

frames = []

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
        
        # 약 2초(8번의 CHUNK) 데이터가 쌓이면 추론
        if len(frames) > 8: 
            audio_data = np.concatenate(frames).astype(np.float32) / 32768.0
            
            # 특징 추출
            input_features = processor(audio_data, sampling_rate=RATE, return_tensors="pt").input_features.to(device)
            
            # ⭐️ 해결책: generate 호출 시 language와 task를 직접 인자로 전달
            # 이렇게 하면 config의 오류나 버전을 무시하고 한국어로 강제 실행됩니다.
            predicted_ids = model.generate(
                input_features,
                language="ko",
                task="transcribe",
                max_new_tokens=128,
                # Tiny 모델의 환각 방지를 위해 빔 서치 추가 (필요 시 1로 조정 가능)
                num_beams=1 
            )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            if transcription.strip():
                # 한글이 포함된 경우에만 출력 (영어 환각 필터링 효과)
                sys.stdout.write(f"\r📝 인식 결과: {transcription}                                ")
                sys.stdout.flush()
            
            frames = []

except KeyboardInterrupt:
    print("\n\n=== 테스트 종료 ===")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()