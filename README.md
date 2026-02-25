- whisper-tiny-local directory 에 모델 설치 필요
- finetuning parameter 은 조절해야함
- train data는 별도 다운로드 ㄱㄱ

실험 1.
실험 목표
10시간 분량의 한국어 군사 용어 데이터를 활용하여 Whisper-tiny 모델의 인식 정확도 향상.
추론(Inference) 단계에서 발생하는 설정 충돌(ValueError, model_kwargs 에러)의 원천 차단.
데이터셋 및 전처리 전략데이터 규모: 약 10시간의 한국어 군사 음성 및 metadata.csv 기반 텍스트 라벨링 데이터.
메모리 관리: librosa를 통한 16kHz 다운샘플링 및 batched=True, keep_in_memory=False 설정을 통한 디스크 캐시 활용.

모델 및 환경 설정
기본 모델: 로컬에 저장된 whisper-tiny-local (Pre-trained).
로딩: model.config 및 model.generation_config에 어떠한 언어 코드나 태스크 정보를 미리 입력하지 않고 순수 가중치만 로드.

학습 하이퍼파라미터
최대 단계: max_steps=500.
배치 사이즈: per_device_train_batch_size=4 및 gradient_accumulation_steps=4.
학습률: 1e-5 (Warmup 1단계 적용).
평가/저장 주기: 100 step 단위로 체크포인트 생성 및 검증 데이터 평가.성공 판정 기준

기술적 지표: 50 step 또는 100 step 저장 시점의 ValueError 발생 여부 (Pass/Fail).
인식 지표: "구수과장" "군수과장" 등 원본 대비 군사 용어 인식률의 유의미한 변화.
설정 유연성: 저장된 모델을 로드하여 기존 비교 코드로 추론할 때 별도의 설정 수정 없이 language="ko" 인자가 정상 작동하는지 여부.