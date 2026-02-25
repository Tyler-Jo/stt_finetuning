import torch
import librosa
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

# 2. 장치 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 모델 로드 중... (장치: {'GPU' if device == 0 else 'CPU'})")

# 3. 모델 및 프로세서 개별 로드 (중복 에러 방지)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH)
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 4. 🔥 핵심: 에러를 유발하는 체크 로직 우회
# pipeline에 language를 넘기지 않고, 모델 설정에 직접 주입합니다.
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="korean", task="transcribe"
)
model.config.suppress_tokens = []

# GenerationConfig를 아예 새로 생성해서 구식 설정을 덮어씁니다.
new_gen_config = GenerationConfig.from_model_config(model.config)
new_gen_config.update(
    language="korean",
    task="transcribe",
    no_timestamps=True,
    forced_decoder_ids=model.config.forced_decoder_ids,
)
model.generation_config = new_gen_config

# 5. 파이프라인 구축
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    device=device,
)

# 6. 오디오 로드 및 추론
test_audio_path = r"./train_data/audio/1.4후퇴.mp3"
print(f"🔊 오디오 로드 중: {test_audio_path}")
audio_array, sampling_rate = librosa.load(test_audio_path, sr=16000)

print("📝 STT 추론 시작...")
# ⚠️ 중요: 여기서 generate_kwargs에 language를 절대 넣지 않습니다!
# 이미 모델이 '한국어'임을 알고 있기 때문입니다.
result = pipe({"raw": audio_array, "sampling_rate": sampling_rate})

print("\n" + "=" * 50)
print(f"🎯 추론 결과: {result['text']}")
print("=" * 50)
