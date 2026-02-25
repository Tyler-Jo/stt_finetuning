import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 1. 설정
model_id = "openai/whisper-tiny"
save_directory = "./whisper-tiny-local" # 모델이 저장될 폴더 경로
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"현재 사용 중인 장치: {device}")

# 2. 모델 및 프로세서 다운로드 (최초 1회 실행 시 로컬에 저장)
print("모델 다운로드 중...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
processor = AutoProcessor.from_pretrained(model_id)

# 로컬 디렉토리에 물리적 저장
model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f"모델이 '{save_directory}' 폴더에 저장되었습니다.")

# 3. 로컬에 저장된 모델 불러와서 테스트하기
print("로컬 저장소에서 모델 로드 중...")    
local_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    save_directory, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
local_processor = AutoProcessor.from_pretrained(save_directory)

# 파이프라인 설정
pipe = pipeline(
    "automatic-speech-recognition",
    model=local_model,
    tokenizer=local_processor.tokenizer,
    feature_extractor=local_processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)

print("준비 완료! 이제 'pipe'를 사용하여 음성 인식을 수행할 수 있습니다.")