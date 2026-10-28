# 펫 3D 모델 파이프라인 API

Spring 서버가 보호소 기록과 사진을 보내면, 파이프라인이 그 개의 3D 모델(GLB)을 만들어 돌려준다.
한 마리에 1.5~4분이 걸리므로 **작업을 등록하고 나중에 결과를 받는 방식**이다.

```
앱 ──(기록+사진)──▶ Spring ──POST /v1/fit──▶ 파이프라인   (선택, 5~20초, 약 $0.013)
                     │      ◀── 적합/주의/부적합 + 실제 크기
                     │
                     ├──POST /v1/jobs──▶ 파이프라인   202 {job_id, status: queued}
                     │                     │ 1.5~4분 (사진 검사 → 읽기 → 텍스처 → 다듬기 → GLB)
                     │◀─ POST callback_url ─┘ 끝나면 작업 JSON 전송 (또는 GET /v1/jobs/{id} 폴링)
                     │
                     ├──GET /v1/jobs/{id}/files/model.glb──▶ GLB 받아서 S3/정적 경로에 저장
                     ▼
앱/AR 페이지 ── three.js GLTFLoader로 GLB 로드, idle 재생, 탭하면 behavior.tap_animation
```

파이프라인 서버 주소는 앱에 노출하지 않는다. Spring이 GLB를 받아 자기 저장소에서 서빙한다.

## 서버 실행 (파이프라인 쪽)

```
cd <파이프라인 루트>
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

| 환경변수 | 뜻 |
|---|---|
| `OPENAI_API_KEY` | 비전·이미지 모델 키. 루트의 `.env`에 있어도 된다 |
| `PIPELINE_API_KEY` | 설정하면 모든 `/v1` 요청에 `X-API-Key` 헤더가 필요하다. 콜백에도 같은 키를 붙여 보낸다 |
| `PIPELINE_CONCURRENCY` | 동시에 돌리는 작업 수, 기본 1. 나머지는 queued로 기다린다 |
| `PIPELINE_PUBLIC_URL` | 응답의 파일 URL 앞에 붙일 주소. 비우면 `/v1/jobs/...` 상대 경로 |
| `BLENDER_EXE` | 미리보기 렌더용 Blender. 없으면 렌더 없이 GLB만 만든다 |
| `PIPELINE_ALLOW_SCRIPTED=1` | 비용 없는 오프라인 대본 실행(`scripted`) 허용. 연동 테스트용 |

필요 파일: 템플릿마다 `output/skeleton_<slug>_small.glb`, `output/templates/<slug>_small/` (jindo, jindo_puppy, maltese,
toypoodle, chihuahua, retriever; git에 없음, 따로 전달). 생김새 분류기 `api/data/template_pick.npz`는 git에 있고, DINOv2 가중치는 처음
쓸 때 Hugging Face에서 받는다(`timm`).
작업 파일은 `output/jobs/<job_id>/`에 남고, 서버를 다시 켜도 끝난 작업은 그대로 조회된다.

## 기록 (record)

폼 필드 `record`에 JSON 문자열로 보낸다. 한국어 키와 영어 키 둘 다 받는다.

| 영어 키 | 한국어 키 | 필수 | 예 | 쓰임 |
|---|---|---|---|---|
| `pet_id` | `관리번호` | O | `"경기-포천-2026-00031"` | 작업·파일 이름 |
| `species` | `종` | O | `"개"` | 개만 된다. 다른 종은 422 |
| `weight_kg` | `몸무게` | O | `2.9` | **실제 크기**(체고)와 판정 |
| `coat_color` | `털색` | O | `"황색"` | 털색 계열 힌트 |
| `age_text` | `추정나이` | | `"2026(60일미만)(년생)"` | 기록용, 템플릿·크기에는 안 씀(강아지는 생김새로 판정). 포인핸드 표기 그대로 |
| `name`, `sex`, `neutered` | `이름`, `성별`, `중성화` | | | 표시용 |

포인핸드 크롤링 데이터에서는 `weight: "2.9(Kg)"`를 숫자로 바꾸고, `age`는 그대로 넣으면 된다.

## POST /v1/fit — 만들어도 되는 개인지 먼저 확인

multipart: `record`, `photo` (jpeg/png/webp, 15MB 이하). 5~20초, 약 $0.013.

```json
{
  "pet_id": "E19",
  "fit": {
    "level": "good", "label": "적합", "score": 100,
    "reasons": [],
    "notes": ["템플릿: 말티즈 (사진과 모델 비교)"],
    "template": "maltese", "picked_by": "vision", "match": "close", "differences": [], "looks": "maltese"
  },
  "size": {"weight_kg": 2.6, "withers_m": 0.219, "scale": 0.354, "rule": "weight"},
  "shape": {"ears": "drop", "coat_length": "long", "leg_length": "normal", "head_size": "large", "...": "..."},
  "photo": {"level": "good", "levels": {"size": "good", "sharpness": "good", "exposure": "good"},
            "short_side_px": 437, "sharpness": 252.0, "clipped_share": 0.015}
}
```

| level | label | 뜻 | 권장 처리 |
|---|---|---|---|
| `good` | 적합 | 템플릿으로 자연스럽게 나온다 | 바로 생성 |
| `fair` | 주의 | 그 개로 보이지만 `reasons`의 차이가 있다 | 생성, 사유를 보여줘도 됨 |
| `poor` | 부적합 | 어느 템플릿으로도 이 개처럼 안 보인다 (포메형·단두형, 단족 등), **또는 사진이 못 쓸 사진** | 생성하지 않거나, 안내 후 사용자가 선택 |

**템플릿** (`fit.template`): 이 개를 만들 모델 — `jindo`, `jindo_puppy`, `maltese`, `toypoodle`, `chihuahua`, `retriever` 중
하나. 나이는 쓰지 않는다(강아지인지도 생김새로 본다). 기준은 "그 모델로 만들어 색을 입히면 그 개로 보이나, 억지스럽지 않나"다.
- 체형 호출(gpt-4o, 3번 투표)에 회색 모델 6개를 A~F로 붙인 시트(`api/model_sheet.py`, 투표마다 순서를 섞음)를 같이 보내, 가장 비슷한
  모델과 그 모델과 **뚜렷이 다른 부위**(귀·털 길이·털 결·주둥이·머리·다리·몸통·꼬리, 진도의 귀·꼬리·체격·주둥이 모프로 바꿀 수 있는 것과
  색·크기는 빼고)를 묻는다(`fit.picked_by: "vision"`). 다른 부위가 없으면 `match: "close"`, 1곳이면 `"rough"`(주의),
  2곳 이상이면 `"poor"`(부적합). `fit.differences`가 그 부위다.
- DINOv2 생김새 분류(`api/template_pick.py`, 로컬)가 `fit.looks`다. `"none"`(포메형·단두형)이면 부적합. 호출이 실패하면 이 분류가 모델을
  고른다(`picked_by: "looks"`).
- 진도 강아지 모델이라도 체중이 7.0 kg(보호소 진도형 2026년생의 95번째 백분위)을 넘으면 진도 성견으로 만든다 — 강아지 메시는 2개월 몸이라
  무거운 개에 쓰면 체고가 16% 작게 나온다. `notes`에 그 사유가 붙는다.
- 단족(다리 짧음, 3표 모두)은 부적합 — 체중으로 낸 크기가 25~35% 틀어진다. 기록 체중이 그 생김새 보호소 개 대부분(5~95%) 밖이면 주의.
- 사람이 직접 고른 30마리(평가셋 20 + 보호소 사진 10, `tools/template_judge.py`)와 비교: 사람이 만들 만하다고 본 22마리 중 20마리에서 같은
  모델(DINOv2만으로는 13), 사람이 억지라고 본 8마리 중 7마리 부적합, 사람이 만들 만하다고 본 22마리 중 2마리 부적합.
- 서버가 처음 판정할 때 DINOv2(약 1.2 GB)를 한 번 읽는다. 모델 시트 이미지(`output/model_sheet/`, `python -m api.model_sheet`)가 서버에
  있어야 한다.

**사진 품질** (`photo`, 호출 없이 로컬 계산): 개 영역의 짧은 변 픽셀(`size`), 선명도(`sharpness`, 흐림·흔들림·작은 사진 확대),
노출(`exposure`, 털의 몇 %가 완전 검정/흰색인지)을 각각 good/fair/poor로 매기고 가장 나쁜 것이 `photo.level`이다.
fair/poor는 `fit.reasons`에 한국어 사유로 들어가고 점수도 깎인다(fair −20, poor −50) — "사진 속 개가 너무 작음(짧은 변 82px): … 개가 크게
나온 사진 필요"처럼 재촬영 안내문으로 바로 보여줄 수 있다. 노출이 poor이면 생성 때 사진 색 측정을 건너뛰고 표 색으로 만든다.
기준: 짧은 변 220px 미만 fair·120px 미만 poor; 노출 클립 20 % 초과 fair·40 % 초과 poor. 평가셋 20장은 전부 good.

`size.withers_m`은 어깨 높이(m)다. 고른 템플릿과 기록 체중으로만 정한다(나이는 쓰지 않음). `rule`은 `"weight"`(견종 실측 기준값) 또는 `"shape"`(실측 기준값이 없는 템플릿, 지금은 `jindo_puppy`: 메시의 등길이·가슴둘레 비율로 낸 k, AI Hub 71520으로 맞춘 식, 견종 하나를 뺀 검증 평균 오차 2.6%, 이때 `k`도 붙는다). 생성된 GLB에 이미 적용된다.
템플릿 견종의 실측 성견(진도: 수컷 50.67 cm · 20.5 kg)에서 체중의 세제곱근에 비례해 정한다. 템플릿과 체형이 같은 개에만 맞으므로
`fit.level`이 `poor`인 개는 크기도 믿을 수 없다.

## POST /v1/jobs — 모델 생성 등록

multipart:

| 필드 | 필수 | 기본 | 뜻 |
|---|---|---|---|
| `record` | O | | 위 기록 JSON |
| `photo` | O | | 전신이 보이는 사진 한 장 |
| `callback_url` | | | 끝나면 작업 JSON을 POST할 주소 |
| `fit_stop` | | `false` | `true`면 부적합일 때 비싼 단계 전에 멈추고 `unsuitable`로 끝낸다 |
| `refine` | | `true` | `false`면 이미지 모델 다듬기 없이 (싸고 빠르지만 경계가 딱딱함) |
| `render` | | Blender 있으면 `true` | 미리보기 PNG 3장 |
| `scripted` | | | 오프라인 대본 이름(`black_shiba`). `PIPELINE_ALLOW_SCRIPTED=1`일 때만 |

응답 `202 Accepted`, `Location: /v1/jobs/{job_id}`:

```json
{"job_id": "E07-a5ca257e", "pet_id": "E07", "status": "queued", "created_at": "2026-09-30T17:21:44+00:00"}
```

## GET /v1/jobs/{job_id} — 상태·결과 (콜백 본문과 같다)

| status | 끝남 | 뜻 | 들어 있는 필드 |
|---|---|---|---|
| `queued` | | 앞 작업을 기다리는 중 | |
| `running` | | 만드는 중 | `started_at` |
| `done` | O | 완성 | 아래 전부 |
| `photo_rejected` | O | 사진이 못 쓰는 사진 (개가 아님, 여러 마리, 사람이 가림, 옷 등) | `reasons` (한국어), `error` |
| `unsuitable` | O | `fit_stop=true`이고 부적합 | `fit` |
| `failed` | O | 서버 오류, 또는 서버 재시작으로 중단 | `error` |

`done` 예:

```json
{
  "job_id": "E07-a5ca257e", "pet_id": "E07", "status": "done",
  "fit": {"level": "good", "label": "적합", "score": 100, "reasons": [], "notes": ["템플릿: 진도 (사진과 모델 비교)"],
          "template": "jindo", "picked_by": "vision", "match": "close", "differences": [], "looks": "jindo"},
  "photo": {"level": "good", "levels": {"size": "good", "sharpness": "good", "exposure": "good"},
            "short_side_px": 333, "sharpness": 254.6, "clipped_share": 0.091},
  "size": {"weight_kg": 25.7, "withers_m": 0.546, "scale": 0.8842, "rule": "weight"},
  "model": {
    "glb_url": "/v1/jobs/E07-a5ca257e/files/model.glb",
    "texture_url": "/v1/jobs/E07-a5ca257e/files/texture.png",
    "renders": {"body": ".../render_body.png", "face": ".../render_face.png", "side": ".../render_side.png"}
  },
  "animations": ["idle", "tail_wag"],
  "behavior": {"playback_rate": 1.0, "tap_animation": "tail_wag", "tap_is_retreat": false},
  "appearance": {"pattern": "tan_point", "base_color": "black", "second_color": "tan",
                 "eye_color": "dark_brown", "features": ["white_chest", "white_paws"]},
  "cost_usd": 0.075, "seconds": 92.1,
  "created_at": "...", "started_at": "...", "finished_at": "..."
}
```

## 파일과 정리

| 요청 | 결과 |
|---|---|
| `GET /v1/jobs/{id}/files/model.glb` | GLB (`model/gltf-binary`), 약 10MB |
| `GET /v1/jobs/{id}/files/texture.png` | 털 텍스처 2048² |
| `GET /v1/jobs/{id}/files/render_side.png` 등 | 미리보기 (render=true일 때) |
| `DELETE /v1/jobs/{id}` | 끝난 작업의 파일 삭제 (204). 진행 중이면 409 |
| `GET /health` | `{"ok", "blender", "openai_key", "running", "queued"}` (키 불필요) |

오류: 401 키 없음/틀림, 403 scripted 비허용, 404 없는 작업/파일, 413 사진 크기, 415 사진 형식, 422 기록 형식·종.

## Spring 쪽

아래 세 파일을 Spring 서버 저장소의 `petmodel` 패키지에 넣어 쓴다.
Spring Boot 4.0.3(webmvc만) 프로젝트에서 컴파일하고, 실제 파이프라인 서버와 판정·조회·다운로드·콜백까지 돌려 확인했다.
Jackson 3(`tools.jackson`) 기준이라 Boot 3 프로젝트면 import를 `com.fasterxml.jackson`으로 바꾼다.

| 파일 | 내용 |
|---|---|
| `PetPipelineDtos.java` | 요청·응답 record (`PetRecord`, `Job`, `Fit`, `Size`, `Model`, `Behavior`, `FitResult`) |
| `PetPipelineClient.java` | `fit()`, `createJob()`, `getJob()`, `download()`, `delete()` |
| `PetModelCallbackController.java` | `POST /internal/pet-models/callback`: 키 확인 → done이면 GLB 받아 저장 (DB 저장은 TODO) |

`application.yml` (`api-key`는 비밀값이니 커밋하지 말고 시크릿이나 서버 `.env`로 넣는다):

```yaml
pet-pipeline:
  base-url: http://pipeline:8000   # 같은 EC2, docker 네트워크 lovablepet의 파이프라인 컨테이너
  api-key: <PIPELINE_API_KEY와 같은 값>
  callback-url: http://<Spring 컨테이너 이름>:8080/internal/pet-models/callback   # 같은 네트워크라 공개 주소가 필요 없다
  model-dir: models            # 콜백이 GLB를 저장할 폴더 (S3로 바꾸면 이 부분 교체)
```

사용 예:

```java
PetRecord rec = new PetRecord("경기-포천-2026-00031", "개", 2.9, "황색", "", "F", false, "2026(60일미만)(년생)");

FitResult check = pipeline.fit(rec, photoBytes, "photo.jpg");
if (check.fit().isPoor()) {
    // 사용자에게 check.fit().reasons() 보여주고 진행 여부 확인
}
Job job = pipeline.createJob(rec, photoBytes, "photo.jpg", false);
// job.jobId()를 펫 행에 저장, 상태 PENDING. 결과는 콜백으로 온다.
// 콜백을 못 받는 환경이면 30초 간격으로 pipeline.getJob(jobId) → isFinished()
```

두 가지 함정 (둘 다 코드에 반영됨):

- **HTTP/1.1 고정**: JDK HttpClient 기본값은 `Upgrade: h2c`로 보내는데, 파이썬 서버(uvicorn)는 그 요청의 본문을 읽지 않아 422 "record/photo missing"이 난다.
- **MultipartBodyBuilder 쓰지 말 것**: Spring 7에서 reactive-streams가 필요해 webmvc 프로젝트에서 `NoClassDefFoundError`가 난다. `LinkedMultiValueMap`에 `HttpEntity` 파트를 담는다.
- `RestClient.Builder` 빈은 Boot 4에서 `spring-boot-starter-restclient`에 있으므로, 이 클라이언트는 `RestClient.builder()`를 직접 쓴다.

## 앱(AR 페이지)에서 쓰기

GLB는 **미터 단위 실제 크기**로 나온다. 체중별 `scale.set(...)`을 따로 하지 말 것 (지금 데모의 소형 0.6/중형 1.0/대형 1.5는 필요 없어진다).

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script>
// pet = Spring이 내려주는 {glbUrl, behavior}
new THREE.GLTFLoader().load(pet.glbUrl, (gltf) => {
  const dog = gltf.scene;                       // 이미 실제 크기, 체형(귀·꼬리·체격)도 적용됨
  scene.add(dog);
  const mixer = new THREE.AnimationMixer(dog);
  mixer.timeScale = pet.behavior.playback_rate;
  const play = (name) => {
    const clip = THREE.AnimationClip.findByName(gltf.animations, name);
    if (clip) mixer.clipAction(clip).reset().play();
  };
  play('idle');
  onTap = () => play(pet.behavior.tap_animation);   // 'tail_wag' 등
  animateHooks.push((dt) => mixer.update(dt));       // 렌더 루프에서 mixer.update(delta)
});
</script>
```

## 비용과 시간 (한 마리)

| 단계 | 시간 | 비용 |
|---|---|---|
| `/v1/fit` | 5~20초 | 약 $0.013 |
| `/v1/jobs` 전체 (다듬기 포함) | 1.5~4분 | 중앙값 약 $0.075 (117회 실측) |
| `refine=false` | 편집 시간만큼 짧음 | 약 $0.03 (비전 호출만, gpt-4o 기준) |

같은 사진·같은 읽기로 다시 만들면 다듬기 편집은 캐시에서 나와 비용이 거의 없다.
