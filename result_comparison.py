import torch
import librosa
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig

# 1. 경로 설정
original_model_path = "./whisper-tiny-local"
finetuned_model_path = "./whisper-tiny-finetuned"
audio_files = ["test_1.m4a", "test_2.m4a", "test_3.m4a"]

device = "mps" if torch.backends.mps.is_available() else "cpu"

def get_transcription(model_path, files):
    print(f"\n[{model_path}] 모델 로딩 및 설정...")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # ⭐️ 해결책: 모든 기존 설정을 무시하고 한국어 전용 설정을 새로 생성
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    
    results = {}
    for audio_path in files:
        if not os.path.exists(audio_path):
            continue
            
        try:
            # 오디오 로드
            audio_array, _ = librosa.load(audio_path, sr=16000)
            input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            
            # ⭐️ 빔 서치(beam search) 적용으로 인식률 상향 및 한국어 고정
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=5,            # 5개의 후보군 중 최적을 선택 (정확도 향상)
                no_repeat_ngram_size=2, # 반복되는 헛소리 방지
                max_length=225
            )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            results[audio_path] = transcription
            
        except Exception as e:
            results[audio_path] = f"에러: {e}"
            
    del model, processor
    if device == "mps": torch.mps.empty_cache()
    return results

# 2. 실행
print("--- [검증] 파인튜닝의 효과를 확인합니다 ---")
original_results = get_transcription(original_model_path, audio_files)
finetuned_results = get_transcription(finetuned_model_path, audio_files)

# 3. 결과 출력
print(f"\n{'='*35} 최종 비교 결과 {'='*35}")
for audio in audio_files:
    print(f"\n[파일명: {audio}]")
    # 원본이 영어 환각을 뱉을 때 파인튜닝이 한국어를 잡는지 확인하세요!
    print(f"  🇰🇷 원본 (Original) : {original_results.get(audio)}")
    print(f"  🔥 파인튜닝 (FT)    : {finetuned_results.get(audio)}")
    print("-" * 75)