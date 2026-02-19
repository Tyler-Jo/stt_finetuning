from transformers import WhisperProcessor

# 원본과 파인튜닝 경로
original_path = "./whisper-tiny-local"
finetuned_path = "./whisper-tiny-finetuned"

# 원본에서 프로세서를 읽어서 파인튜닝 폴더에 다시 저장 (누락된 json 생성)
processor = WhisperProcessor.from_pretrained(original_path)
processor.save_pretrained(finetuned_path)
print(f"✅ {finetuned_path}에 설정 파일 복사 완료!")
