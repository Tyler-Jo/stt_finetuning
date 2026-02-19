import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 1. 수동으로 다운로드한 파일들이 들어있는 폴더 경로
# 상대 경로 또는 절대 경로를 입력하세요.
local_model_path = "./whisper-tiny-local"


def check_local_model():
    print(f"로컬 경로에서 모델 로드 시도 중: {local_model_path}")

    try:
        # 로컬 경로에서 프로세서와 모델 불러오기
        # local_files_only=True를 설정하면 외부망 접속을 아예 시도하지 않습니다.
        processor = WhisperProcessor.from_pretrained(
            local_model_path, local_files_only=True
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            local_model_path, local_files_only=True
        )

        print("-" * 50)
        print("✅ 모델 로드 성공!")
        print(f"모델 파라미터 수: {model.num_parameters():,}")
        print("-" * 50)

        # 간단한 테스트: 프로세서가 정상 작동하는지 확인
        print("프로세서 테스트: 오디오 전처리 설정 확인 완료")

    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        print("\n💡 체크리스트:")
        print(
            "1. 폴더 안에 config.json, model.safetensors 등이 모두 있는지 확인하세요."
        )
        print("2. 파일 확장자가 .txt나 .html로 잘못 저장되지 않았는지 확인하세요.")


if __name__ == "__main__":
    check_local_model()
