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

## 환경변수 (`.env`)

| 이름 | 뜻 |
|---|---|
| `OPENAI_API_KEY` | 비전·이미지 모델 키 |
| `PIPELINE_API_KEY` | 설정하면 모든 `/v1` 요청에 `X-API-Key` 헤더가 필요 |
| `PIPELINE_CONCURRENCY` | 동시 작업 수, 기본 1 (uvicorn 워커는 하나로 둔다 — 작업 목록이 프로세스 메모리에 있다) |
| `PIPELINE_PUBLIC_URL` | 응답 파일 URL 앞에 붙일 주소 |
| `BLENDER_EXE` | 미리보기 렌더용 Blender |
| `PIPELINE_ALLOW_SCRIPTED=1` | 오프라인 대본 실행 허용 (연동 테스트용) |

## 템플릿 자산 갱신

```
tar -czf dist/tpl_<slug>.tar.gz output/skeleton_<slug>_small.glb output/templates/<slug>_small
tar -czf dist/model_sheet.tar.gz output/model_sheet
```

Release에 올리고 서버의 `~/lovablepet-ai`에서 `gh release download <태그> -D /tmp/a && for f in /tmp/a/*.tar.gz; do tar -xzf "$f"; done`
후 `docker compose restart`.
