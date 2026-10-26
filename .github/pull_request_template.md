<!--
제목: type(scope): 설명   예) feat(fit): 단족 판정 추가 · fix(face): 흰 가슴이 안 읽힘 · chore(deploy): ...
작업 브랜치(feat/ fix/ refactor/ docs/ chore/) → dev.  배포는 dev → main.  hotfix/ → main, 머지 후 dev에도 반영.
-->

## 무엇을 바꿨나

-

## 왜

<!-- 문제, 요청, 측정 결과. 관련 이슈가 있으면 #번호 -->

## 종류

- [ ] feat 기능
- [ ] fix 버그 수정
- [ ] perf 속도·메모리
- [ ] refactor 동작 변경 없는 정리
- [ ] docs 문서
- [ ] chore / ci 설정·의존성·배포 스크립트
- [ ] 배포 (dev → main)

## 확인한 것

<!-- 실제로 돌린 것만 체크하고 결과를 적는다. 안 돌렸으면 이유를 적는다. -->

- [ ] CI 통과 (서버 이미지 빌드 + `/health`)
- [ ] 대본 실행 (호출·비용 0):
      `python -m api.run --record tests/record_black.json --photo tests/fixtures/image.png --out output/runs/<이름> --scripted black_shiba`
- [ ] 로컬 테스트 (`python tests/test_<이름>.py`):
- [ ] 평가셋 (`python -m tests.run_evalset --tag <태그>`): 판독·텍스처가 바뀌면 고친 한 마리가 아니라 20마리 전부로 본다
  - 결과 / 비용: <!-- 예) 20마리 중 18 개선·2 같음, $1.40 -->
- [ ] 바뀐 결과 사진 (전 / 후 렌더)

## 영향

<!-- 해당하는 것만 체크 -->

- [ ] API 계약 변경 (`docs/API.md`) — Spring 저장소도 고쳐야 함
- [ ] 환경변수 추가·변경 — README 표와 서버 `~/lovablepet-ai/.env`
- [ ] 템플릿 자산 변경 — 새 Release 태그
- [ ] 배경 제거 Lambda 변경 (`lambda/rembg/`, `api/bria.py`) — main 머지 때 함수가 새 이미지로 바뀜
- [ ] 메모리·시간·API 비용이 달라짐 (전 → 후):
- [ ] 없음

## 배포 메모 (dev → main일 때)

<!-- 머지하면 서버가 재시작되고 진행 중이던 작업은 failed가 된다. 작업이 없을 때 머지. -->

- 서버에 먼저 넣을 값:
- 머지 후 확인할 것:
