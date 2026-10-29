# Lovable Pet – 파이프라인 서버

보호소 기록(종·체중·털색)과 사진을 받아, 리깅·애니메이션이 살아 있는 템플릿 GLB에 그 개의 코트를 입혀 돌려준다.
Spring 서버가 HTTP로 부른다 — 계약은 [docs/API.md](docs/API.md).

## 폴더

```
api/              파이프라인 (기본 lite: api/lite.py), HTTP 서버 api/server.py, CLI python -m api.run
api/data/         생김새 분류기 template_pick.npz
ar_pipeline/      템플릿 제작 Blender 스크립트 (드라이버 new_template.py)
Dockerfile        서버 이미지 (CPU torch, api/, tests/scripts.py)
deploy/           EC2용 docker-compose.yml
lambda/rembg/     배경 제거(BRIA RMBG 2.0) Lambda 이미지 (handler.py, Dockerfile, 메모리를 줄인 모델 slim_model.py; 계산은 api/bria.py)
docs/             API.md (Spring 연동 계약; Spring 쪽 클라이언트 코드는 Spring 서버 저장소에 있다)
tests/scripts.py  오프라인 대본 (PIPELINE_ALLOW_SCRIPTED=1, 호출·비용 없음)
```

템플릿 자산은 git에 없다(약 1.2 GB): `output/skeleton_<slug>_small.glb`, `output/templates/<slug>_small/`, `output/model_sheet/`
(slug: jindo, jindo_puppy, maltese, toypoodle, chihuahua, retriever). GitHub Release `assets-v1`에 tar.gz로 있고,
프로젝트 루트에서 풀면 제자리에 들어간다.

## 로컬 실행

```
pip install torch torchvision          # GPU가 있으면 CUDA 빌드
pip install -r requirements.txt
# Release assets-v1의 *.tar.gz를 프로젝트 루트에서 tar -xzf
# .env: OPENAI_API_KEY=sk-...
uvicorn api.server:app --port 8000
python -m api.run --record <기록.json> --photo <사진> --out output/runs/<이름>
```

Blender(`BLENDER_EXE`)가 있으면 미리보기 PNG도 렌더하고, 없으면 GLB만 만든다.

## 도커로 실행

```
docker build -t lovablepet-ai .
docker run -p 8000:8000 --env-file .env -v ./output:/app/output -v ./cache:/cache lovablepet-ai
```

## 배포 (AWS EC2 + GitHub Actions)

`.github/workflows/deploy.yml`: dev·main으로의 PR과 push마다 이미지를 빌드하고 `/health`로 뜨는지 확인한다.
main에 머지되면 이미지를 GHCR(`ghcr.io/lovable-pet/lovablepet-ai-ar`)에 올리고 EC2에서 새 이미지로 교체한다.
첫 배포 때 서버에 템플릿이 없으면 Release `assets-v1`에서 받아 푼다.

서버 준비 (한 번):

1. EC2: Ubuntu 24.04, t3.large(8 GB), 디스크 30 GB, Elastic IP. 보안그룹 22(SSH), Spring 포트. 8000은 열지 않는다.
   8 GB는 배경 제거를 Lambda로 보낼 때(`REMBG_URL`, 아래 "배경 제거 Lambda") 기준이다. 서버가 직접 돌리면 그 한 번에 약 8 GB가 더 든다.
2. 서버에서:
   ```
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 gh
   sudo usermod -aG docker $USER          # 다시 로그인해야 적용
   mkdir -p ~/lovablepet-ai && nano ~/lovablepet-ai/.env   # 아래 환경변수
   ```
3. 저장소 Actions secrets: `SSH_HOST`(Elastic IP), `SSH_USER`(ubuntu), `SSH_KEY`(서버 `authorized_keys`에 넣은 키의 개인키).

서버 폴더 `~/lovablepet-ai/`: `docker-compose.yml`(배포가 복사), `.env`, `output/`(템플릿 + 작업 파일), `cache/`(DINOv2·rembg 가중치).
Spring 컨테이너는 외부 네트워크 `lovablepet`에 붙어 `http://pipeline:8000`으로 부른다.
재배포하면 진행 중이던 작업은 failed가 되므로 작업이 없을 때 main에 머지한다.

로그: `docker logs -f pipeline`

## 배경 제거 Lambda

배경 제거(BRIA RMBG 2.0, rembg 기본 모델)는 1024×1024 한 번에 CPU 메모리 약 8 GB를 쓴다. 작업 하나 동안 서버 최고치 10.5 GB 중 이 단계가
대부분이었고, 나머지 단계는 3 GB대다. `REMBG_URL`이 있으면 이 계산만 Lambda(`lambda/rembg/`)가 하고 서버는 사진을 1024×1024로 줄여 보내
마스크를 받는다. rembg의 계산을 둘로 나눈 것이다(`api/bria.py`).

Lambda 이미지는 같은 모델을 메모리를 덜 쓰게 고쳐 넣는다(`lambda/rembg/slim_model.py`, 이미지 빌드 때 변환). 8 GB는 모델이 아니라 ONNX 변환
방식 탓이었다. opset 14 파일이라 변형 합성곱(DeformConv) 20개가 각각 GatherND 4개짜리 노드 60개로 풀려 있었다. 이것을 ONNX 기본 DeformConv로
바꾸고, 큰 DeformConv는 입력 채널을 나눠 더하고, 1024×1024 마지막 단계(768 MB 확대 + 960 MB 이어 붙이기)는 1×1 Conv를 확대보다 먼저 한다.
모두 같은 계산을 순서만 바꾼 것이다. 여기에 glibc가 큰 블록을 바로 돌려주게 하는 설정(Dockerfile의 `MALLOC_*`)을 더했다.

2026-10-30 확인:
- 마스크: 평가셋·픽스처 29장에서 원래 모델과 최대 1/255 차이, 0.5 기준 IoU 1.000000.
- Lambda 이미지를 Docker로 함수와 같은 제한(메모리 3008 MB, CPU 2개)에 두고 5회 연속: 최고 1,886 MiB, 한 장 약 29초(첫 호출 35초).
  같은 조건에서 원래 모델은 메모리 부족으로 죽는다.
- 실제 Lambda(3000 MB): 한 장 54초(모델 읽기 포함), 다음 호출 43초. Max Memory Used 2,871 MB는 모델 파일의 페이지 캐시(약 1 GB)를
  Lambda가 사용 메모리로 세서다. Docker에서 캐시를 비우고 재면 2,870 MiB로 같다. 그래서 handler가 모델을 읽은 뒤 캐시를 버린다
  (`posix_fadvise`, 같은 조건에서 2,138 MiB). 새 이미지 뒤 첫 호출 하나는 180초 제한에 걸렸다 — 제한 시간 5분.

2026-10-27 확인 (서버, 원래 모델로 잰 값):
- 서버: 대본 작업 두 개 연속 최고 3.1 GB(이전 10.5 GB), 작업 67~73초(이전 126초). 키가 틀리면 Lambda가 401.
- 얼굴 자르기는 배경 제거를 다시 돌리지 않고 사진 준비의 컷 상자를 쓴다(작업당 모델 1회). 29장 중 28장이 다시 돌렸을 때와 0~8 px 차이,
  `images (1)`은 바닥 목줄이 상자에 들어가 72 px 넓다.

`.github/workflows/rembg-lambda.yml`: `lambda/rembg/`나 `api/bria.py`가 main에 바뀌면(또는 수동 실행) 이미지를 ECR에 올리고 함수를 새 이미지로 바꾼다.

AWS 준비 (한 번, 서울 리전 ap-northeast-2):

1. ECR: 프라이빗 리포지토리 `lovablepet-rembg`.
2. IAM 사용자(GitHub Actions용)에 아래 정책을 붙이고 액세스 키를 만들어 저장소 secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`에 넣는다.
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
       {"Effect": "Allow",
        "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                   "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
        "Resource": "arn:aws:ecr:ap-northeast-2:<계정 ID>:repository/lovablepet-rembg"},
       {"Effect": "Allow",
        "Action": ["lambda:GetFunction", "lambda:GetFunctionConfiguration", "lambda:UpdateFunctionCode"],
        "Resource": "arn:aws:lambda:ap-northeast-2:<계정 ID>:function:lovablepet-rembg"}
     ]
   }
   ```
3. Actions에서 "rembg Lambda"를 수동 실행한다. 함수가 아직 없으니 이미지만 ECR에 올라간다(태그 = 커밋 SHA).
4. Lambda → 함수 생성 → 컨테이너 이미지, 이름 `lovablepet-rembg`, 위 이미지, x86_64. 구성:
   메모리 3008 MB(새 계정 상한, 이것으로 충분), 제한 시간 5분(`api/bria.py` `TIMEOUT_S`와 같게), 환경 변수 `REMBG_KEY`(`openssl rand -hex 32`), 함수 URL(인증 유형 NONE),
   예약된 동시성 3(URL이 공개라 남용 상한).
5. 서버 `.env`에 `REMBG_URL=<함수 URL>`, `REMBG_KEY=<같은 값>`을 넣고 `docker compose up -d`.

이후 main에 머지할 때 이 파일들이 바뀌어 있으면 함수가 자동으로 새 이미지로 바뀐다.
비용: 한 번에 약 2.9 GB × 30~60초 = 90~180 GB-초, Lambda 무료 한도(월 40만 GB-초) 안에서 월 2,200~4,400회.

## 환경변수 (`.env`)

| 이름 | 뜻 |
|---|---|
| `OPENAI_API_KEY` | 비전·이미지 모델 키 |
| `PIPELINE_API_KEY` | 설정하면 모든 `/v1` 요청에 `X-API-Key` 헤더가 필요 |
| `PIPELINE_CONCURRENCY` | 동시 작업 수, 기본 1 (uvicorn 워커는 하나로 둔다 — 작업 목록이 프로세스 메모리에 있다) |
| `PIPELINE_PUBLIC_URL` | 응답 파일 URL 앞에 붙일 주소 |
| `BLENDER_EXE` | 미리보기 렌더용 Blender |
| `PIPELINE_ALLOW_SCRIPTED=1` | 오프라인 대본 실행 허용 (연동 테스트용) |
| `REMBG_URL` | 배경 제거 Lambda의 함수 URL. 없으면 서버가 rembg의 원래 모델로 직접 돌린다(한 번에 약 8 GB) |
| `REMBG_KEY` | Lambda와 같은 비밀 값 (`X-Rembg-Key` 헤더) |

## 템플릿 자산 갱신

```
tar -czf dist/tpl_<slug>.tar.gz output/skeleton_<slug>_small.glb output/templates/<slug>_small
tar -czf dist/model_sheet.tar.gz output/model_sheet
```

Release에 올리고 서버의 `~/lovablepet-ai`에서 `gh release download <태그> -D /tmp/a && for f in /tmp/a/*.tar.gz; do tar -xzf "$f"; done`
후 `docker compose restart`.
