# Lovable Pet – 파이프라인 서버

보호소 기록(종·체중·털색)과 사진을 받아, 리깅·애니메이션이 살아 있는 템플릿 GLB에 그 개의 코트를 입혀 돌려준다.
Spring 서버가 HTTP로 부른다 — 계약은 [docs/API.md](docs/API.md), 애니메이션 재생은 [docs/ANIMATION.md](docs/ANIMATION.md).

## 폴더

```
api/              파이프라인 (기본 lite: api/lite.py), HTTP 서버 api/server.py, CLI python -m api.run,
                  아이폰 AR용 GLB → USDZ api/usdz.py
api/data/         생김새 분류기 template_pick.npz
ar_pipeline/      템플릿 제작·수정 스크립트 (드라이버 new_template.py)
Dockerfile        서버 이미지 (CPU torch, api/, tests/scripts.py)
deploy/           EC2용 docker-compose.yml
lambda/rembg/     배경 제거(BRIA RMBG 2.0) Lambda 이미지
docs/             API.md (Spring 연동 계약), ANIMATION.md (클립과 재생)
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
아이폰 AR(Quick Look)용 USDZ는 서버가 GLB 옆에 같이 만든다(`model.usdz`). 따로 만들려면
`python -m api.usdz <GLB> [--clip <클립> | --all] [-o <출력>]`.

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
   8 GB는 배경 제거를 Lambda로 보낼 때(`REMBG_URL`) 기준이다. 서버가 직접 돌리면 그 한 번에 약 8 GB가 더 든다.
2. 서버에서:
   ```
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 gh
   sudo usermod -aG docker $USER          # 다시 로그인해야 적용
   mkdir -p ~/lovablepet-ai && nano ~/lovablepet-ai/.env   # 아래 환경변수
   ```
3. 저장소 Actions secrets: `SSH_HOST`(Elastic IP), `SSH_USER`(ubuntu), `SSH_KEY`(서버 `authorized_keys`에 넣은 키의 개인키).

서버 폴더 `~/lovablepet-ai/`: `docker-compose.yml`(배포가 복사), `.env`, `output/`(템플릿 + 작업 파일), `cache/`(모델 가중치).
Spring 컨테이너는 외부 네트워크 `lovablepet`에 붙어 `http://pipeline:8000`으로 부른다.
재배포하면 진행 중이던 작업은 failed가 되므로 작업이 없을 때 main에 머지한다.

- 로그: `docker logs -f pipeline`
- `.env`를 고친 뒤엔 `docker compose up -d` (`restart`는 `.env`를 다시 읽지 않는다)
- 배포가 실행 중인 이미지에 `:latest` 태그를 붙여 두어 손으로 띄워도 같은 이미지를 쓴다. 그 태그가 없으면 compose가 GHCR에서 받으려다 `unauthorized`로 멈춘다

## 배경 제거 Lambda

배경 제거는 한 번에 CPU 메모리 약 8 GB를 쓴다. `REMBG_URL`이 있으면 이 계산만 Lambda(`lambda/rembg/`, 메모리를 줄인 모델)가 하고
서버는 사진을 1024×1024로 줄여 보내 마스크를 받는다(`api/bria.py`).
`.github/workflows/rembg-lambda.yml`: `lambda/rembg/`나 `api/bria.py`가 main에 바뀌면(또는 수동 실행) 이미지를 ECR에 올리고 함수를 새 이미지로 바꾼다.

AWS 준비 (한 번, 서울 리전 ap-northeast-2):

1. ECR: 프라이빗 리포지토리 `lovablepet-rembg`.
2. IAM 사용자(GitHub Actions용)에 아래 정책을 붙이고 액세스 키를 저장소 secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`에 넣는다.
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
4. Lambda → 함수 생성 → 컨테이너 이미지, 이름 `lovablepet-rembg`, 위 이미지, x86_64.
   메모리 3008 MB, 제한 시간 5분, 환경 변수 `REMBG_KEY`(`openssl rand -hex 32`), 함수 URL(인증 유형 NONE), 예약된 동시성 3.
5. 서버 `.env`에 `REMBG_URL=<함수 URL>`, `REMBG_KEY=<같은 값>`을 넣고 `docker compose up -d`.

## 환경변수 (`.env`)

| 이름 | 뜻 |
|---|---|
| `OPENAI_API_KEY` | 비전·이미지 모델 키 |
| `PIPELINE_API_KEY` | 설정하면 모든 `/v1` 요청에 `X-API-Key` 헤더가 필요 |
| `PIPELINE_CONCURRENCY` | 동시 작업 수, 기본 1 (uvicorn 워커는 하나로 둔다 — 작업 목록이 프로세스 메모리에 있다) |
| `PIPELINE_PUBLIC_URL` | 응답 파일 URL 앞에 붙일 주소 |
| `BLENDER_EXE` | 미리보기 렌더용 Blender |
| `PIPELINE_ALLOW_SCRIPTED=1` | 오프라인 대본 실행 허용 (연동 테스트용) |
| `REMBG_URL` | 배경 제거 Lambda의 함수 URL. 없으면 서버가 직접 돌린다(한 번에 약 8 GB) |
| `REMBG_KEY` | Lambda와 같은 비밀 값 (`X-Rembg-Key` 헤더) |

## 템플릿 자산 갱신

부위 마스크를 다시 자를 때는 이 순서로 돌리고 앵커를 다시 렌더한다:

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

필요할 때 돌리는 수정 (자세한 이유·방법은 각 스크립트 맨 위 설명):

- UV가 다른 곳 텍셀을 읽는 삼각형: `python -m ar_pipeline.fix_uv_overlaps --slug <slug>` (이미 만든 개는 `--glb <GLB>`)
- 브라우저에서 흑백 경계에 번개무늬: `python -m ar_pipeline.mip_islands --glb <만든 개 GLB> --slug <slug>`
- 눈·코 보호 영역만 다시: `python ar_pipeline/build_landmarks.py <glb> <템플릿 폴더> <geometry> --protect-only`
- 흰 털 위 어두운 털 조각: `python -m ar_pipeline.fix_normals --slug <slug>`
- 홍채 없는 회색 눈: `python -m ar_pipeline.paint_eyes --slug <slug> --install` (홍채는 있고 반사광만 없으면 `--lights-only`)
- 검은 원판·털뿐인 발바닥: `python -m ar_pipeline.paint_pads --slug <slug> [--ref jindo] --install`
- 흰 털에 박힌 검은 줄·점: `python -m ar_pipeline.fill_gutters --slug <slug> --install`
- 스무딩으로 녹은 가슴 되돌리기: `python -m ar_pipeline.restore_geometry --slug <slug> --from <이전 GLB>` 뒤 `fix_normals`
- 늘어진 귀에 뺨 색: `python -m ar_pipeline.ear_flaps --slug <slug> --install`
- 수염을 따로 재질로 (맨 마지막): `python -m ar_pipeline.split_whiskers --slug <slug>`
- 눈 둘레 마스크만: `python -m ar_pipeline.open_face_masks --slug <slug> --eye-ring-only`

애니메이션 (순서: `rig_fix` → `mouth` → `pose_clips`; 클립은 docs/ANIMATION.md):

```
python -m ar_pipeline.rig_fix --slug <slug> --install
python -m ar_pipeline.mouth --slug <slug> --install          # 짖기용 입 (진도)
python -m ar_pipeline.mouth --slug <slug> --remake --install # 이미 있는 입을 걷어내고 다시 (소형견 템플릿: mouth.MUZZLES)
python -m ar_pipeline.pose_clips --slug <slug> --install      # 기본 클립
python -m ar_pipeline.pose_clips --slug <slug> --clip bark --install
python -m ar_pipeline.pose_clips --slug <slug> --clip lying --clip sit_down --clip stand_up --clip lie_down --clip get_up --install
python -m ar_pipeline.mouth_clips --slug <slug> --install     # 이미 있는 클립에 헥헥거리는 입만 얹기 (몸 동작은 그대로)
```

죽은 척·자세 전환의 모션캡처(`ar_pipeline/data/dog_clips.npz`, AI4Animation SIGGRAPH 2018)는 **CC BY-NC 4.0: 출처 표시, 상업적 이용 불가**.

배포용 묶음:

```
tar -czf dist/tpl_<slug>.tar.gz output/skeleton_<slug>_small.glb output/templates/<slug>_small
tar -czf dist/model_sheet.tar.gz output/model_sheet
```

Release에 올리고 서버의 `~/lovablepet-ai`에서 `gh release download <태그> -D /tmp/a && for f in /tmp/a/*.tar.gz; do tar -xzf "$f"; done`
후 `docker compose restart`.
