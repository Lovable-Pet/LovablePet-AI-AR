# 개발 안내

파이프라인을 직접 돌리거나 템플릿을 고칠 때 보는 문서입니다. 서비스 소개는 [README](../README.md)에 있습니다.

## 템플릿 자산

템플릿 자산은 git에 없다(약 1.2 GB): `output/skeleton_<slug>_small.glb`, `output/templates/<slug>_small/`, `output/model_sheet/`
(slug: jindo, jindo_puppy, maltese, toypoodle, chihuahua, retriever). GitHub Release(`assets-v3` 등)에 tar.gz로 있고,
프로젝트 루트에서 풀면 제자리에 들어간다.

## 로컬 실행

```
pip install torch torchvision          # GPU가 있으면 CUDA 빌드
pip install -r requirements.txt
# Release(deploy.yml의 ASSETS_TAG)의 *.tar.gz를 프로젝트 루트에서 tar -xzf
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

## 환경변수 (`.env`)

| 이름 | 뜻 |
|---|---|
| `OPENAI_API_KEY` | 비전·이미지 모델 키 |
| `OPENAI_MODEL` | 사진을 읽는 모델, 기본 `gpt-6-luna` (`gpt-4o`로 두면 예전 모델) |
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

`SHA256.txt`(`(cd dist && sha256sum *.tar.gz) > dist/SHA256.txt; sha256sum output/skeleton_*_small.glb >> dist/SHA256.txt`)와 함께 새 태그의 Release에 올리고, `deploy.yml`의 `ASSETS_TAG`를 그 태그로 바꿔 main에 머지.
