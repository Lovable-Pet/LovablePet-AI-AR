# Lovable Pet – 파이프라인 서버

보호소 기록(종·체중·털색)과 사진을 받아, 리깅·애니메이션이 살아 있는 템플릿 GLB에 그 개의 코트를 입혀 돌려준다.
Spring 서버가 HTTP로 부른다 — 계약은 [docs/API.md](docs/API.md).

## 폴더

```
api/              파이프라인 (기본 lite: api/lite.py), HTTP 서버 api/server.py, CLI python -m api.run,
                  아이폰 AR용 GLB → USDZ api/usdz.py
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

아이폰 AR(Quick Look)용 USDZ는 서버가 GLB 옆에 같이 만든다(`model.usdz`, 대기 루프 클립: 진도 `standing`, 그 외 `idle`;
Quick Look 안에서는 명령·음성이 안 된다 - `docs/ANIMATION.md` 6장). 따로 만들려면
`python -m api.usdz <GLB> [--clip <클립> | --all] [-o <출력>]` (Blender 없이 `usd-core`만 쓴다):
뼈대·스킨·클립은 키 시각마다 그대로 옮기고(STEP·CUBICSPLINE은 그 사이를 채움), 모프는 기본 가중치로 굽고,
Quick Look이 무시하는 정점 색(진도의 입 안)은 삼각형마다 8 px 칸에 구운 텍스처로 바꾼다.

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

`.env`를 고친 뒤엔 서버에서 `docker compose up -d`(`restart`는 `.env`를 다시 읽지 않는다). 배포가 실행 중인 이미지에 `:latest` 태그를 붙여 두어서
손으로 띄워도 같은 이미지를 쓴다. 그 태그가 없으면 compose가 GHCR에서 `:latest`를 받으려다 `unauthorized`로 멈춘다.

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

부위 마스크를 다시 자를 때(생성 모델 템플릿)는 이 순서로 돌리고 앵커를 다시 렌더한다:

```
python ar_pipeline/refine_masks.py --template <slug>_small --glb output/skeleton_<slug>_small.glb \
    --geometry output/templates/<slug>_small/geometry --label-smooth 14 --edge-clean 8
python ar_pipeline/refine_face_masks.py --template <slug>_small --glb output/skeleton_<slug>_small.glb \
    --geometry output/templates/<slug>_small/geometry
python -m ar_pipeline.open_face_masks --slug <slug>
python -m ar_pipeline.smooth_masks --slug <slug>
python -m ar_pipeline.tidy_regions --slug <slug>
python ar_pipeline/render_anchors.py --template <slug>_small
```

새 템플릿을 만든 뒤 한 번: UV가 다른 곳 섬에 놓인 삼각형을 제자리 표면으로(`python -m ar_pipeline.fix_uv_overlaps --slug <slug>`;
이미 돌린 템플릿에 꼭짓점만 걸친 부분 겹침을 고칠 때는 `--partial-only`),
눈·코 보호 영역만 다시(`python ar_pipeline/build_landmarks.py <glb> <템플릿 폴더> <geometry> --protect-only`).
털 조각이 흰 털 위 어두운 삼각형으로 보이면 접힌 조각의 법선을 바로잡기(`python -m ar_pipeline.fix_normals --slug <slug>`; 언제 돌려도 됨).
눈이 홍채 없는 회색 원판이면(생성 모델에 따라) 홍채·동공·반사광을 그려 넣기(`python -m ar_pipeline.paint_eyes --slug <slug> --install`; 원본 눈에 한 번만, 토이푸들 11-02 적용).
발바닥이 통째로 검은 원판이거나 털뿐이면 다른 템플릿의 패드를 옮겨 그리기(`python -m ar_pipeline.paint_pads --slug <slug> [--ref jindo] --install`; 백업 `work/backup_pads_<slug>/`).
흰 털에 검은 줄·점이 박혀 보이면 UV 여백 채우기(`python -m ar_pipeline.fill_gutters --slug <slug> --install`; 백업 `work/backup_gutters_<slug>/`).
가슴 형상 스무딩(`smooth_geometry --regions chest`)은 짧은 털 템플릿(진도·치와와)용이다 — 곱슬 토이푸들에 돌렸더니 가슴 곱슬이 녹아 매끈한 판이 됐다. 되돌리려면 그 전 GLB에서 움직인 정점만 복원하고 법선을 다시(`python -m ar_pipeline.restore_geometry --slug <slug> --from <이전 GLB>` 뒤 `fix_normals`; 토이푸들 11-02 적용).
애니메이션을 넣을 템플릿은 먼저 리그를 고친다(`python -m ar_pipeline.rig_fix --slug <slug> --install`; 늘 백업 `work/backup_weights_<slug>/`(auto_rig 원본)에서 시작, 진도 11-03 적용): auto_rig는 관절을 키의 고정 비율에 두고(엉덩이 관절이 등선, 무릎이 허벅지 가운데, 목 관절이 목 뒤 표면, 머리 회전축이 정수리 뒤) 피부를 거리 가우시안으로 나눠, 앉으면 허벅지가 접히고 고개를 돌리면 얼굴이 찌그러지고 목에 주름 고리가 생기고 몸통을 비틀면 어깨에 이음선이 생겼다. → 다리 관절은 메시의 가슴 바닥 높이로(`refine_masks.leg_levels`; 뒷다리 무릎·발목은 다리 단면 한가운데로, auto_rig는 허벅지 뒤 표면에 붙여 앉으면 무릎 앞 피부가 개구리 다리처럼 옆으로 펼쳐졌다), 목·머리 관절은 목 단면 중심으로(머리 축은 단면이 머리로 바뀌는 곳); 다리는 각자 뼈 사슬로(뒷다리는 뼈마다 높이 구간을 맡고 관절에서 좁게 섞으며, 배·엉덩이와의 경계는 넓게: 앉으면 허벅지가 배에 대해 약 65° 돈다; 앞다리도 같은 방식으로, 위팔이 옆 가슴을 덜 가져가게(손을 들면 가슴이 따라오던 것)), 몸통·목·머리는 엉덩이→등→가슴→목→머리 사슬을 따라, 얼굴 전체는 머리 하나에, 꼬리는 꼬리 사슬(몸통 사슬보다 가까운 피부)로, 귀 뼈·숨쉬기 뼈 몫은 유지.
애니메이션 클립 6개(서기·앉기·손·구르기·죽은 척·꼬리 흔들기; 숨쉬기만 auto_rig 것)는 템플릿 몸에 맞춰 만든다(`python -m ar_pipeline.pose_clips --slug <slug> --install`; 원본 `work/backup_anim_<slug>/`, 진도 11-03 적용, 확인 페이지 `output/anim3d.html`): **서기**는 템플릿의 쉬는 자세 그대로(idle처럼) 예전 시바 원본(`archive/shiba`)의 고개·귀·꼬리·몸 기울기 잔동작(시바는 한쪽만 보므로 뒤 12초는 좌우를 뒤집어 반대쪽을 본다, 24초 반복), **앉기**는 뼈 길이로 푼 자세(등은 곧게 41°, 골반은 거의 말지 않아 엉덩이를 낮게(엉덩관절 6 cm; 더 말면 꼬리 뿌리 아래 늘어진 털을 깔고 앉아 엉덩이가 뜸), 목은 82°까지만 세워 가슴이 목 앞으로 튀어나오지 않게, 머리 관절로 얼굴 수평, 뒷다리는 허벅지를 40°로 세워 무릎이 옆구리 쪽에, 정강이는 바닥으로, 발은 앞으로, 꼬리는 엉덩이 왼쪽으로 감아 바닥에; 고개는 서기처럼 좌우 번갈아, 24초 반복), **손**은 영상에서 뽑은 동작(`ar_pipeline/video_motion.py` → `ar_pipeline/data/video_clips.npz`; Pexels 영상을 MoCapAnything(MIT)으로 3D화)을 앉기 위에(든 앞발은 영상에서 올리고 내리는 때만 받아 빠르게 들고 내밀고 있다가 내리며, 모양은 어깨부터 드는 손 주기 자세 `OFFER_LEG`로; 짚은 다리·어깨뼈·꼬리는 앉기 그대로, 몸 흔들림은 30 %로), **구르기**는 엎드린 자세에서 다리를 접고 시바 원본의 구르기 각도·타이밍대로(엎드린 자세는 엉덩이를 한쪽으로 눕힌 편한 엎드림이되 가슴·앞다리·고개는 정면으로 반듯하게 - 11-06: 캡처 그대로는 가슴이 10° 기울고 9° 돌아가 왼앞발이 가슴 밑으로 넘어가 있었음, 사용자 "엎드려 구르기할때 발이랑 몸, 고개가 비틀려있는데"; 반듯한 스핑크스에서 구르게 바꿨더니 딱딱해지고 뒷다리가 이상해져 되돌림; 구르는 동안 뒷다리는 엎드렸을 때만큼 접어 무릎 약 130°·뒤꿈치 100°로 발이 허벅지 밑에 - 전엔 무릎 90~120°·뒤꿈치 55~65°라 뒤집힐 때 뒷다리가 펴져 뻣뻣하게 뻗쳤음, 사용자 "구르기 뒷발좀 자연스럽게 … 뒤틀리지않았나"; 11-06 두 번째: 클립은 엎드려 있기와 같은 반듯한 스핑크스로 시작해 0.6초에 엉덩이를 옆으로 눕히고 다리를 접어 구른 뒤 다시 0.6초에 반듯하게 돌아와 끝남(5초, 구르기 자체는 그대로 엉덩이를 눕힌 엎드림에서) - 전엔 엉덩이를 눕히고 뒷발 하나가 옆으로 뒤집힌 엎드림으로 6초를 있어 정면에서 발이 왼쪽으로 뒤틀려 보였음, 사용자 "엎드려구르기 여전히 왼쪽으로 발이 뒤틀려있고"; 엉덩이를 눕힌 엎드림의 뒷다리는 뒤꿈치·발이 무릎과 같은 평면에서 접히게(`hinge_leg`, 12° 이내 - 캡처 그대로는 깔고 누운 왼뒷다리의 뒤꿈치가 옆으로 64° 꺾여 발이 배 밑으로 돌아가 있었음, 사용자 "왼쪽다리 뭔가 꺾이는거 개선못해? 둘다 그런데"; 바닥에 닿은 발은 납작하게 - 발바닥이 쉬는 자세처럼 바닥을 보고 몸이 향한 쪽을 가리키게(`flat_paw`; 리그의 발 뼈는 발바닥이 평평할 때 앞발 44°·뒷발 59° 아래를 향하는데, 엎드린 자세들은 거의 수평으로 눕혀 발바닥이 앞으로 45~70° 들려 기본 카메라(앞 왼쪽)에서 왼발 발바닥이 보이고 꺾여 보였음, 사용자 "여전히 이상하잖아 직접봐 왼쪽다리") - 발끝은 그 자리에 두고 발허리뼈를 손목·뒤꿈치 높이를 지키게 기울이고 위 다리는 IK로 따라감; 엎드려 있기·구르기의 스핑크스와 엉덩이 눕힌 엎드림 모두; 11-06 세 번째: 엉덩이 눕힌 엎드림은 모캡 그대로 약 15°만 눕힘 - 25°를 더 눕혀 가슴 대비 39° 비틀었더니 구르기 앞뒤로 허리가 꺾여 보이고 위쪽 뒷다리가 솟아 발바닥이 보였음, 사용자 "엎드려구르기랑 죽은척 초반에 뒷다리 그냥 골반이 왼쪽으로 틀어져있는것같은데" - 엎드려 있기의 스핑크스는 확인한 그대로 25° 눕힌 엎드림에서 만듦(`SPHINX_DOWN_DEG`; 새 엎드림에서 만들면 꼬리 44°·오른쪽 엉덩이 12°가 달라져 엎드려 있기와 구르기 첫 프레임이 어긋남); 구르는 동안 뒷발은 발등뼈에서 발가락을 30~35° 오므려(서 있을 때보다 58~63°) 늘어뜨림 - 서 있을 때처럼 앞으로 펴 두니 발바닥이 개의 앞을 보고 있어서 오른쪽으로 넘어갔다 배로 돌아올 때 앞 왼쪽 카메라에서 왼쪽 뒷발이 발바닥을 보이며 들렸음, 사용자 "엎드려 구르기할때 왼쪽 발바닥 보이게 왜 드는거야자꾸?"; 구르기 앞뒤 엎드림에서 두 뒷다리는 스핑크스처럼 접은 채 엉덩이만 눕힘(`fold_hind`: 무릎은 옆구리, 뒤꿈치는 엉덩이 옆 바닥, 발등·발은 앞으로) - 모캡 다리는 눕지 않은 쪽으로 빼 놓은 자세라, 깔리는 왼쪽 뒷다리는 발목이 118°로 펴져 뒤꿈치가 7 cm 들린 Z자로 서 있고 길이축으로 42° 돌아가 있었고, 오른쪽 뒷발은 엉덩이 밑에서 10 cm 옆으로 미끄러져 나와 5 cm 들려 있어서, 구르기 직전에 뒷발을 옆으로 뺐다 넣고 뒤꿈치를 드는 것처럼 보였음, 사용자 "맨처음에 걍 엎드려있다가 왼쪽 뒷다리 들고 도는데 왼쪽 뒷다리가 기괴하다니까?", "구르기 시작하기 직전에 발을 왼쪽으로 잠깐 뺀다니까 이거 근데 기괴하게 생겼다고"), **죽은 척**은 실제 개 모션캡처의 연속 동작(누워 있는 동안은 제자리에, 누운 다리는 모캡처럼 바닥에 내려놓음; 11-06: 옆으로 눕기 전까지 몸의 옆 기울기(캡처는 10~16°)를 펴고, 골반도 가슴 뒤로 반듯하게(`square_hips`; 캡처는 서서 엎드릴 때까지 엉덩이가 왼쪽으로 최대 18° 휘고 12° 비틀려 있었음 - 11-06 세 번째, 같은 사용자 말; 기울기처럼 옆으로 넘어가는 만큼 덜, 시간축으로 부드럽게), 각 발을 몸 중심선 자기 쪽에 - 평소 간격 쪽으로 당기되 캡처의 변화 35 %는 남기고 0.6~1.5배 안에(서 있을 때 왼뒷발이 중심선을 넘고 일어설 때 앞다리가 X자로 엇갈렸음), 가슴이 들려 있는 동안 디딘 앞발이 바닥에 닿게 엉덩이를 축으로 몸을 숙임(앉을 때 앞발이 8~13 cm 떠서 발바닥이 뒤집혔음; 어깨가 서 있을 때 높이 아래로 - 엉덩이가 낮은 동안은 그 95 %까지 - 내려올 때까지, 최대 40°, 고개는 숙이지 않고 들고 있던 대로), 디딘 앞발은 앉고 엎드리고 일어나는 동안에도 고정 - 사용자 "죽은척할때도 다리가 살짝 부자연스럽게 비틀리는데"; 11-06 두 번째(사용자 "죽은척도 왼발이 내려가다가 튀는느낌임 그리고 중간에 경련와서 부자연스러움"): 캡처의 발이 바닥 3 cm 안에 있는 동안은 미끄러지고 있어도 붙잡고 그 자리를 캡처 발이 미끄러진 만큼 옮김(멈춘 발만 붙잡다 놓으니 일어날 때 템플릿의 긴 다리가 바닥에서 5 cm 떠올랐음), 붙잡고 놓는 정도는 약 0.4초에 걸쳐 S자로, 닿지 않는 자리는 바닥을 따라 다리 쪽으로 당김, 숙인 몸에 바닥을 파고든 허공의 다리는 그 다리만 어깨·엉덩이에서 들어 올림(전엔 몸 전체를 들어 디딘 앞발까지 3~6 cm 떴음), 붙잡은 발도 몸 중심선을 넘지 않게(캡처 개는 서서 오른뒷발을 몸 아래로 7 cm 넘김), 허공의 발은 자기 쪽으로 조금씩 따라가게, 다리 회전은 시간축으로 살짝 부드럽게 한 뒤 붙잡은 발을 다시 자리에(매 프레임 IK·보정이 프레임마다 달라 다리가 떨렸음), 이어 붙이는 마지막 0.5초에서도 붙잡은 자리를 함께 섞음(거기서 발이 1~2 cm 튀었음); 네 다리 모두 무릎(팔꿈치) 아래 뼈가 무릎이 접히는 평면에서 12° 이상 벗어나지 않게(`hinge_leg` - 캡처를 입힌 그대로는 왼쪽으로 누울 때·일어날 때 왼뒷다리 뒤꿈치가 옆으로 꺾이고 발이 배 밑으로 돌아갔고(최대 74°), 왼앞다리 손목도 옆으로 꺾였음(최대 63°), 사용자 "왼쪽다리 뭔가 꺾이는거"), 옆으로 누웠다 일어날 때 밑에 깔린 앞다리는 끝까지 바닥에 내려놓음(옆으로 누운 앞다리 모양이 풀리는 동안 6 cm 떴음); 바닥에 닿은 발은 납작하게(`flat_paw`, 캡처 발이 바닥 3~6 cm 안인 만큼, 옆으로 누운 동안은 빼고, 시간축으로 부드럽게 - 앉을 때 왼뒷발 발바닥이 앞을 보고 왼앞발이 접혀 디뎠음; 발이 거의 수직일 때 발 자체 방향을 따르니 0.4초 동안 70° 바깥으로 돌았다 돌아와서 몸 방향을 따름), 다리 시간 스무딩 1.6프레임, 허공의 발은 매 프레임 60 %씩 자기 쪽으로(일어설 때 왼앞발이 몸 아래로 디디며 오른앞다리와 X자로 엇갈렸음), 발은 몸 중심선에서 평소 간격의 0.8배 안으로는 들어오지 않게; 옆으로 누워 있는 동안 앞다리는 가슴에서 앞으로 나란히 뻗고 위쪽 다리를 아래쪽 위에 조금 앞으로 얹음(`SIDE_FORELEGS`, 몸이 50~70° 넘어가는 만큼) - 캡처는 위쪽 앞다리가 손목에서 아래쪽 위로 엇갈리고 발이 말려 발바닥이 위를 봤음, 사용자 "죽은척할때 중간에 앞발 비틀리는거만 개선"; 템플릿 자세를 이어 붙여 새로 만든 버전은 딱딱해 되돌림; `ar_pipeline/dog_mocap.py` → `ar_pipeline/data/dog_clips.npz`, AI4Animation SIGGRAPH 2018 데이터, **CC BY-NC 4.0: 출처 표시, 상업적 이용 불가**). **꼬리 흔들기**는 서기 자세에서 꼬리를 들고 모양 그대로 좌우로 활처럼 흔들며 엉덩이도 반대로 살짝(4초 반복; auto_rig 것은 꼬리가 비틀리며 말려 올라갔다). 모든 클립에 idle과 같은 숨쉬기(숨쉬기 뼈 세로 3 %, 약 4초)가 들어간다. 새 묘기 영상을 넣으려면 WSL에 MoCapAnything을 한 번 설치하고(`ar_pipeline/mocapanything/setup_wsl.sh`) 영상을 돌려 BVH를 받은 뒤(`ar_pipeline/mocapanything/run_video.sh`) `video_motion --clip <이름> --bvh ...`. 영상은 개 한 마리가 옆에서, 몸 전체가 보이고, 카메라가 고정된 것이 좋다(정면이나 배를 보이고 누운 자세는 위아래를 헷갈린다).
짖기(bark)는 입이 열리는 템플릿에만 만든다(진도 11-05 적용, 확인 페이지 `output/anim3d.html`의 짖기): 생성 메시는 입이 한 면으로 닫혀 있고(입술선은 주름일 뿐) auto_rig의 jaw_jnt는 피부가 없어서, 먼저 입을 연다(`python -m ar_pipeline.mouth --slug <slug> --install`; 백업 `work/backup_mouth_<slug>/`, 입이 이미 있으면 그 백업에서 다시) - 입술선(입꼬리 주름에서 코 밑 가운데를 지나 반대 입꼬리까지 안으로 접힌 모서리)을 따라 정점을 둘로 갈라 아래 입술을 새 복사본에(위치·UV·법선·모프 그대로라 닫힌 입은 전과 같음), jaw_jnt를 입꼬리 1.8 cm 뒤·1.2 cm 위 가운데(턱관절)로 옮기고, 피부는 표면을 따라 아래 입술이 위 입술보다 가까운 만큼 턱에(턱 뒤 목구멍과 턱관절 위는 빼고), 생성 메시의 아래턱은 윗입술 밑 1~1.5 cm 쐐기라 아랫면을 5 mm 내려 두껍게(입술 가장자리는 그대로; 털 파이프라인은 꼭짓점 2 cm 안의 텍셀을 쓰므로 구운 지도는 그대로 맞음), 입 안(검은 입술·잇몸·이빨·입천장·혀·목구멍 벽)을 정점 색 재질 `mouth`의 두 번째 프리미티브로 - 아래 잇몸은 입술선 위까지 솟은 둔덕(닫히면 윗입술 안에 들어감), 이빨은 잇몸 마루에 개의 치열대로(한쪽마다 앞니 3·송곳니·작은어금니 4·큰어금니 1) 둥근 원뿔; 입술 가장자리는 피부와 같은 가중치, 모프는 가까운 피부에서, 쉬는 자세와 모프 7개 각각에서 피부 밖으로 나온 정점은 안쪽으로 당김. 털 입히기(api/glb.py)는 텍스처가 있는 coat만 다시 칠하므로 입 색은 그대로 간다. 그다음 `python -m ar_pipeline.pose_clips --slug <slug> --clip bark --install`: 서기 자세에서 고개를 살짝 들고 두 번 짖고 쉬는 3초 반복, 한 번에 입 28°(두 번째는 26°) 0.29초(실제 개의 짖음 0.25~0.35초, Yin & McCowan 2004; 30°는 얇은 아래턱이 오리 부리 같았음), 두 번 사이엔 2°(4°는 이빨이 보여 으르렁 느낌), 이 클립만 60 fps(24 fps는 입이 한 프레임에 16° 열려 툭 끊겼음), 짖을 때마다 몸도 같이 - 직전에 숨 들이쉬며(가슴 6 %) 몸을 살짝 뒤로·고개를 숙였다가, 짖을 때 목·고개를 앞·위로 튀기고(코 약 4 cm, 주둥이 13°) 가슴을 2 cm 앞·아래로 던져 앞다리로 버티며 배를 조이고, 짖은 뒤 가슴이 가라앉고 고개를 끄덕이며 돌아옴(전에는 머리·가슴이 0.3~0.6 cm만 움직여 몸 전체가 보이면 입만 열리는 것 같았음); 두 번째 짖음은 몸 동작 80 %에 고개를 왼쪽으로 6° 돌려서, 귀는 짖기 전 앞으로·짖을 때 뒤로, 꼬리는 흔듦. 순서는 rig_fix → mouth → pose_clips (rig_fix --install은 입 없는 auto_rig 원본에서 시작하므로 그 뒤엔 mouth부터 다시).
자세 전환 클립(인터랙션용; 진도 11-05 적용, 확인 페이지 `output/anim3d.html`의 명령 버튼)과 엎드려 있기 루프: `python -m ar_pipeline.pose_clips --slug <slug> --clip lying --clip sit_down --clip stand_up --clip lie_down --clip get_up --install` (엎드린 자세가 바뀌면 rollover도 함께; 11-06 전에 만든 GLB엔 `--drop lie_flop --drop lie_square`로 없앤 전환 둘을 뺌) - `sit_down` 서기→앉기, `stand_up` 앉기→서기, `lie_down` 서기→엎드려 있기, `get_up` 엎드려 있기→서기, `lying`은 반듯한 스핑크스 자세(뒷다리를 몸 아래 좌우 똑같이 접고 앞다리는 앞으로; `crouch()` - 구르기 시작 자세에서 엉덩이 눕힘과 몸 비틀림을 빼고 뒷다리를 앉기처럼 접되 허벅지를 앞으로 15° 아래로)에서 숨쉬고 귀·꼬리만 구르기의 엎드린 동안처럼 움직이는 6초 루프. 각 전환은 앞 루프의 첫 프레임에서 시작해 뒤 루프의 첫 프레임에서 정확히 끝남(앱은 루프 → 전환 1회 → 다음 루프; 앉기↔엎드리기는 서기를 거쳐, "굴러"는 엎드려 있기 → rollover(같은 자세에서 시작해 끝남) → 엎드려 있기; 재생 중 들어온 명령은 그 클립이 끝난 뒤에). 루프 클립을 다시 만들면 전환도 다시 만든다(같은 실행에서 만들면 그 결과에, 아니면 GLB에 든 루프에 맞춤). 처음엔 모캡 개의 관절 회전을 그대로 입혔더니 앉을 때 뒷다리가 개구리처럼 벌어지고 앞다리가 엇갈리고 몸이 모캡 개처럼 비틀려서(사용자 11-05: "다리가 전체적으로 부자연스럽게 꺾여서… 몸도 왜 삐딱하게"), 지금은 템플릿의 자기 자세로 만든다: 몸통은 앞 자세 모양에서 뒤 자세 모양으로 좌우 대칭인 채로(돌거나 기울지 않음), 엉덩이·어깨 높이는 모캡(`dog_mocap.TRANSITIONS` - D1_013 명령에 따라 제자리에서 앉고 엎드리고 일어나는 개, 엎드렸다 일어날 땐 D1_061 제자리에서 앞발을 당겨 일어나는 개)의 높이 변화를 두 자세 높이에 맞춰(앉을 땐 엉덩이 먼저, 엎드릴 땐 앞쪽 먼저, 일어날 땐 앞쪽이 살짝 먼저; 엎드릴 땐 앞발이 나간 만큼의 60 % 이상 어깨도 내려가게 - 모캡 높이만 따르니 몸을 높이 든 채 곧은 앞다리만 멀리 뻗었다가 떨어졌음), 몸은 엉덩이를 축으로 기울여 어깨를 그 높이에; 고개는 두 자세의 고개 사이에서(목이 나눠 받음); 다리는 키 포즈처럼 IK로 - 발은 제자리에 있다가 모캡 개가 그 발을 움직이는 구간에 다음 자리로 디딤(모캡이 움직인 거리만큼, 마지막 걸음은 뒤 자세 자리로; 걸음은 최소 0.29초, 4 cm 들고 앞발은 발목을 뒤로 접으며), 모캡이 안 움직이는 발은 몸이 85 % 바뀐 뒤 한 걸음씩(같은 쪽 발이 들려 있으면 그 뒤에); 무릎은 허벅지가 가리키는 방향(두 자세 사이에서 돌림) 쪽으로 굽혀서 반쯤 앉은 자세에서도 무릎이 앞으로(굽힘 방향 자체를 보간하면 앉기의 옆으로 나온 무릎 때문에 개구리 다리), 팔꿈치는 두 자세의 팔꿈치 방향 사이에서(위팔 방향으로 정하면 앞발을 뻗을 때 앞으로 꺾임); 디딘 발과 내딛는 발이 닿지 않으면 엉덩이를 축으로 몸을 숙임(필요한 숙임을 앞뒤 2프레임까지 유지한 뒤 부드럽게 - 부드럽게만 하니 한 프레임에 필요한 숙임이 반으로 줄어 앞다리가 닿지 않는 자리로 한 프레임에 곧게 펴졌다 다시 굽어 경련처럼 보였음, 사용자 11-06 "중간에 경련와서"; 숙여서 가슴이 바닥에 들어가면 몸을 올림), 걸음의 들기와 앞발 접기는 sin²로 천천히 시작해 천천히 끝냄(sin은 첫 프레임부터 최고 속도라 다리가 툭 움직였음); 꼬리는 앞 자세에서 뒤로 곧게 폈다가 뒤 자세로(앉기에서 바로 보간하면 허벅지를 뚫음), 매 프레임 조금씩만 돌려 길이 방향으로 꼬이지 않게; 숨쉬기는 넣지 않음. 모캡 원본(AI4Animation `MotionCapture.zip`, 33 MB, https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip)을 `work/dog_mocap_bvh/`에 풀면 `python -m ar_pipeline.dog_mocap --bvh work/dog_mocap_bvh`로 `ar_pipeline/data/dog_clips.npz`를 다시 만들 수 있음(기존 standing·play_dead는 그대로 나옴).
늘어진 귀 템플릿에서 뺨 마스크가 귀 앞면까지 덮여 있으면(뺨 색이 귀에 칠해짐) 귀 뼈 스킨 가중치로 귀를 찾아 머리(skull)로 넘기기(`python -m ar_pipeline.ear_flaps --slug <slug> --install`; 원본 마스크는 `work/backup_earflaps_<slug>/`, 리트리버 11-02 적용).
맨 마지막에(`shape_keys` 뒤) 수염 가닥을 따로 재질로(`python -m ar_pipeline.split_whiskers --slug <slug>`; 개마다 수염색 하나, `whisker_pad.png`도 만듦).
위 순서를 다시 돌리지 않은 템플릿에 눈 둘레 마스크(`_eye_ring.png`)만 만들려면 `python -m ar_pipeline.open_face_masks --slug <slug> --eye-ring-only`.

```
tar -czf dist/tpl_<slug>.tar.gz output/skeleton_<slug>_small.glb output/templates/<slug>_small
tar -czf dist/model_sheet.tar.gz output/model_sheet
```

Release에 올리고 서버의 `~/lovablepet-ai`에서 `gh release download <태그> -D /tmp/a && for f in /tmp/a/*.tar.gz; do tar -xzf "$f"; done`
후 `docker compose restart`.
