# 애니메이션 사용법 (앱 / AR 페이지)

파이프라인이 만든 GLB 하나에 개의 동작 클립이 모두 들어 있다. 앱은 GLB를 불러서
**지금 자세의 루프를 틀어 두고, 명령(터치·음성)이 오면 전환 클립을 한 번씩 재생한 뒤 다음 루프로** 넘어가면 된다.
참고 구현: `output/anim3d.html`(model-viewer 버전, `python -m http.server 8765 --directory output` 후 열기).

## 1. 어떤 개에 어떤 클립이 있나

| 템플릿 | 클립 |
|---|---|
| 진도 (`skeleton_jindo_small`) | 13개 전부 (아래 표) |
| 몰티즈, 리트리버, 토이푸들, 치와와, 진도 강아지 | `idle`, `tail_wag` 두 개뿐 |

클립 목록은 작업 결과 JSON의 `animations`(Spring `Job.animations()`)나 로드한 `gltf.animations`의 이름으로 확인한다.
`sit_down`이 없으면 명령 기능을 끄고 `idle` + 탭 `tail_wag`만 쓴다.

## 2. 클립 표 (이름은 대소문자 그대로)

**루프 = 자세(상태)**

| 클립 | 길이 | 상태 | 내용 |
|---|---|---|---|
| `standing` | 24초 | 서기 | 서서 숨 쉬며 고개를 좌우로 둘러봄 (기본 대기) |
| `sitting` | 24초 | 앉기 | 앉아서 숨 쉬며 둘러봄 |
| `lying` | 6초 | 엎드리기 | 반듯하게 엎드려(스핑크스) 숨 쉬고 꼬리 살랑 |
| `idle` | 4초 | 서기 | 숨쉬기만 (진도 외 템플릿의 기본 대기) |

**전환 = 한 번 재생** (앞 루프의 첫 프레임에서 시작해 뒤 루프의 첫 프레임에서 정확히 끝남)

| 클립 | 길이 | 시작 → 끝 |
|---|---|---|
| `sit_down` | 1.29초 | 서기 → 앉기 |
| `stand_up` | 1.58초 | 앉기 → 서기 |
| `lie_down` | 1.71초 | 서기 → 엎드리기 |
| `get_up` | 1.46초 | 엎드리기 → 서기 |

앉기 ↔ 엎드리기 직접 전환은 없다. 서기를 거친다(`stand_up` → `lie_down`).

**묘기**

| 클립 | 길이 | 어디서 | 재생 | 비고 |
|---|---|---|---|---|
| `rollover` | 5초 | 엎드리기 | 1회 | 엎드린 자세에서 시작해 굴러서 같은 자세로 끝남 |
| `play_dead` | 13.12초 | 서기 | 1회 | 털썩 넘어져 옆으로 누웠다가 일어남. 실제 개 모션캡처 |
| `bark` | 3초 | 서기 | 2회 | 두 번 짖고 쉼 (60 fps) |
| `tail_wag` | 4초 | 서기 | 1회 | 탭 반응 (`behavior.tap_animation`) |
| `shake` | 5.79초 | 앉기 | 루프 | 앞발 내밀기 |

## 3. 명령 흐름

```
            sit_down              lie_down
   앉기 ◀──────────── 서기 ────────────▶ 엎드리기
        ────────────▶      ◀────────────
            stand_up              get_up
```

| 명령 | 음성 예 | 할 일 |
|---|---|---|
| 앉아 | 앉아 | 앉기로 이동 → `sitting` 루프 |
| 엎드려 | 엎드려 | 엎드리기로 이동 → `lying` 루프 |
| 일어나 | 일어나, 일어서 | 서기로 이동 → `standing` 루프 |
| 손 | 손, 악수 | 앉기로 이동 → `shake` 루프 |
| 굴러 | 굴러 | 엎드리기로 이동 → `rollover` 1회 → `lying` 루프 |
| 짖어 | 짖어, 멍멍 | 서기로 이동 → `bark` 2회 → `standing` 루프 |
| 빵! | 빵, 빵야 | 서기로 이동 → `play_dead` 1회 → `standing` 루프 |
| (탭) | - | 서기일 때만 `tail_wag` 1회 → `standing` 루프 |

"이동"은 지금 상태에서 목표 상태까지 전환 클립을 차례로 한 번씩 재생하는 것이다(앉기에서 "엎드려"면 `stand_up` → `lie_down`).

규칙:

- **루프는 언제 끊어도 된다.** 바로 다음 클립으로 넘어간다(크로스페이드 0.25초).
- **한 번 재생 중인 클립(전환·묘기)은 끝까지 재생한다.** 중간에 끊으면 반쯤 앉은 자세에서 다음 클립 첫 자세로 튄다.
  그동안 들어온 명령은 그 클립이 끝난 뒤 실행하고, 여러 개 들어왔으면 마지막 것만 실행한다.
- 상태는 전환이 **끝났을 때** 바꾼다.

## 4. three.js 구현 (r128, `docs/API.md`의 로딩 코드와 같은 방식)

주의할 점:

- 새 클립을 틀 때 이전 액션을 멈추거나 페이드아웃해야 한다. `clipAction(clip).reset().play()`만 하면
  이전 클립도 계속 재생돼 두 동작이 섞인다.
- 한 번(또는 n번) 재생은 `setLoop(THREE.LoopRepeat, n)` + `clampWhenFinished = true`. 끝나면 mixer의 `'finished'` 이벤트가 온다.
- 전환 클립은 다음 루프의 첫 프레임에서 끝나므로 이어지는 루프는 처음(시간 0)부터 틀면 이음새가 없다.
- `play_dead`, `shake`, `tail_wag`, `bark`의 첫 프레임은 루프의 첫 프레임과 조금 다르다(다리·꼬리 12~84°).
  그래서 페이드를 0.3~0.4초 준다(아래 `FADE`).

```js
const LOOP = { stand: 'standing', sit: 'sitting', down: 'lying' };
const MOVES = { sit_down: ['stand', 'sit'], stand_up: ['sit', 'stand'],
                lie_down: ['stand', 'down'], get_up: ['down', 'stand'] };
const FADE = { default: 0.25, play_dead: 0.4, shake: 0.3 };       // 초
const COMMANDS = {
  '앉아':   [['to', 'sit']],
  '엎드려': [['to', 'down']],
  '일어나': [['to', 'stand']],
  '손':     [['to', 'sit'], ['hold', 'shake']],
  '굴러':   [['to', 'down'], ['play', 'rollover']],
  '짖어':   [['to', 'stand'], ['play', 'bark', 2]],
  '빵':     [['to', 'stand'], ['play', 'play_dead']],
};

class DogAnimator {
  constructor(gltf, playbackRate = 1) {
    this.mixer = new THREE.AnimationMixer(gltf.scene);
    this.mixer.timeScale = playbackRate;
    this.clips = {};
    for (const c of gltf.animations) this.clips[c.name] = c;
    this.hasCommands = Object.keys(MOVES).every((n) => this.clips[n]);
    this.state = 'stand';
    this.current = null;
    this.busy = null;   // 한 번 재생 중인 클립의 Promise
    this.run = 0;       // 가장 최근 명령 번호 (이전 명령을 취소하는 데 씀)
    this.loop(this.hasCommands ? 'standing' : 'idle');
  }

  update(dt) { this.mixer.update(dt); }   // 렌더 루프에서 매 프레임

  start(name, times, fade) {              // times: 0이면 루프, 아니면 그 횟수만큼
    const action = this.mixer.clipAction(this.clips[name]);
    action.reset();
    action.setLoop(THREE.LoopRepeat, times || Infinity);   // 횟수가 끝나면 'finished'
    action.clampWhenFinished = !!times;                    // 끝난 자리(마지막 프레임)에 멈춰 있게
    action.setEffectiveWeight(1);
    action.play();
    if (this.current && this.current !== action) {
      if (fade > 0) this.current.crossFadeTo(action, fade, false);
      else this.current.stop();
    }
    this.current = action;
    return action;
  }

  loop(name, fade = FADE[name] ?? FADE.default) { this.start(name, 0, fade); }

  once(name, times = 1, fade = FADE[name] ?? FADE.default) {
    const p = new Promise((resolve) => {
      const action = this.start(name, times, fade);
      const done = (e) => {
        if (e.action !== action) return;
        this.mixer.removeEventListener('finished', done);
        resolve();
      };
      this.mixer.addEventListener('finished', done);
    });
    this.busy = p;
    p.then(() => { if (this.busy === p) this.busy = null; });
    return p;
  }

  route(to) {                             // 지금 상태 → to 까지의 전환 클립들 (BFS)
    const seen = { [this.state]: [] }, todo = [this.state];
    while (todo.length) {
      const at = todo.shift();
      if (at === to) return seen[at];
      for (const [clip, [from, next]] of Object.entries(MOVES)) {
        if (from === at && !(next in seen)) { seen[next] = [...seen[at], clip]; todo.push(next); }
      }
    }
    return [];
  }

  async command(word) {
    const steps = COMMANDS[word];
    if (!steps || !this.hasCommands) return;
    const mine = ++this.run;
    while (this.busy) await this.busy;            // 재생 중인 전환·묘기는 끝까지
    if (mine !== this.run) return;                // 그사이 더 새 명령이 왔으면 이건 버림
    for (const [kind, what, times] of steps) {
      if (kind === 'hold') { this.loop(what); return; }
      const clips = kind === 'to' ? this.route(what) : [what];
      for (const clip of clips) {
        // 전환에서 이어지는 클립은 같은 자세에서 시작하므로 짧게만 섞는다
        await this.once(clip, times || 1, clip in MOVES ? FADE.default : undefined);
        if (clip in MOVES) this.state = MOVES[clip][1];
        if (mine !== this.run) return;
      }
    }
    this.loop(LOOP[this.state], 0.1);
  }

  async tap(tapAnimation = 'tail_wag') {          // behavior.tap_animation
    if (this.busy || !this.clips[tapAnimation]) return;
    if (this.hasCommands && this.state !== 'stand') return;
    const mine = ++this.run;
    await this.once(tapAnimation, 1, 0.3);
    if (mine === this.run) this.loop(this.hasCommands ? LOOP[this.state] : 'idle', 0.3);
  }
}

// 사용
new THREE.GLTFLoader().load(pet.glbUrl, (gltf) => {
  scene.add(gltf.scene);                          // 미터 단위 실제 크기 그대로 (스케일하지 말 것)
  const dog = new DogAnimator(gltf, pet.behavior.playback_rate);
  animateHooks.push((dt) => dog.update(dt));
  onTap = () => dog.tap(pet.behavior.tap_animation);
  onCommand = (word) => dog.command(word);        // 버튼·음성에서 '앉아' 등
});
```

model-viewer로도 같은 흐름을 만들 수 있다(`output/anim3d.html`의 `start`·`once`·`perform`). 다만 두 가지를 꼭 지켜야 한다.

- **`play({repetitions})`는 이름을 바꾼 뒤 `await mv.updateComplete` 다음에 부른다.** model-viewer는 새 `animationName`을
  다음 업데이트(지금 코드가 끝난 뒤)에 무한 반복으로 튼다. 그래서 `mv.animationName = 'sit_down'; mv.play({repetitions: 1})`처럼
  붙여 쓰면 횟수가 덮어써져 `'finished'`가 영영 오지 않는다. 첫 명령 뒤 버튼이 전부 먹통이 되던 원인이 이것이다.
- **한 번 재생한 클립은 마지막 프레임에 멈춘 채 남는다.** 다음에 다시 틀면 바로 끝나 버린다. 그래서 전환이 끝나 루프로 넘어갈 때
  `mv.currentTime = 0`(모든 클립을 처음으로)을 해 둔다. 이 리셋은 이름을 바꾸기 **전에** 해야 한다(크로스페이드가 이 시계로 돈다).
  넘어갈 때 크로스페이드는 0으로 한다. 전환의 마지막 프레임이 루프의 첫 프레임과 같아서 이음새가 없고, 리셋된 전환 클립이 페이드아웃되며
  첫 자세로 번쩍이지도 않는다. `bark`·`play_dead`는 끝 자세가 루프와 조금 달라서 300ms로 넘어간다(둘 다 시작=끝이라 번쩍이지 않음).

```js
async function start(mv, clip, {times = 0, fade = 300, reset = false} = {}) {
  mv.animationCrossfadeDuration = fade;
  if (reset) mv.currentTime = 0;                        // 이름을 바꾸기 전에
  mv.animationName = clip;
  await mv.updateComplete;                              // 새 이름을 (무한 반복으로) 튼 다음에
  mv.play(times ? {repetitions: times} : undefined);    // 횟수
}
// 앉아: 루프에서 전환을 한 번 (지금 루프 자세에서 300ms 크로스페이드) → 끝나면 앉기 루프로 바로
await new Promise((done) => { mv.addEventListener('finished', done, {once: true}); start(mv, 'sit_down', {times: 1}); });
await start(mv, 'sitting', {reset: true, fade: 0});
```

## 5. 음성 명령 (Web Speech API)

```js
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const WORDS = [[/앉아|앉자/, '앉아'], [/엎드려|엎드리/, '엎드려'], [/일어나|일어서/, '일어나'],
               [/^\s*손\s*$|악수/, '손'], [/굴러|구르/, '굴러'], [/짖어|멍멍/, '짖어'], [/빵/, '빵']];
const rec = new SR();
rec.lang = 'ko-KR';
rec.continuous = true;
rec.onresult = (e) => {
  const text = e.results[e.results.length - 1][0].transcript;
  const hit = WORDS.find(([re]) => re.test(text));
  if (hit) dog.command(hit[1]);
};
micButton.onclick = () => rec.start();   // 사용자 동작으로 시작해야 하고, https와 마이크 권한이 필요하다
```

- 지원: Chrome·Edge·Android Chrome, iOS Safari 14.5 이상(`webkitSpeechRecognition`). 없으면 명령 버튼만 보여 준다.
  아이폰은 페이지 안 3D 뷰에서만 된다. AR(Quick Look) 안에서는 안 된다(6장).
- "손"은 다른 말에도 자주 들어가므로(손님 등) 한 단어로 말했을 때만 받는다.

## 6. 아이폰 (AR Quick Look)

iOS Safari에는 WebXR이 없어서 아이폰의 AR 버튼은 애플의 AR Quick Look을 띄우고, Quick Look은 서버가 만든 USDZ(`usdz_url`)를
연다([API.md](API.md#아이폰-usdz)). Quick Look은 페이지와 따로 도는 뷰어라 그 안에서는 이 문서의 코드가 돌지 않고,
파일에 든 대기 루프 하나(진도 `standing`, 그 외 `idle`)를 반복할 뿐이다.

| 아이폰 화면 | 대기 루프 | 탭 | 명령 버튼 | 음성 |
|---|---|---|---|---|
| 페이지 안 3D 뷰 (three.js / model-viewer) | ✅ | ✅ | ✅ | ✅ iOS Safari 14.5 이상 |
| AR (Quick Look) | ✅ | ❌ | ❌ | ❌ |

- 아이폰에서는 명령·음성을 3D 뷰에서 받고, AR은 "내 방에 둔 개 보기"로 쓴다. AR 버튼 옆에 "AR에서는 명령을 못 해요" 정도를 적어 두면 좋다.
- AR을 닫으면 페이지로 돌아온다. AR은 열 때마다 대기 루프부터 시작한다.
- 자세별 USDZ를 만들어 지금 자세로 AR을 여는 방식은, 자세를 바꿀 때마다 AR을 닫았다 다시 열어야 해서 쓰지 않는다.
- Quick Look 자체는 USDZ에 넣은 애플의 탭 규칙(preliminary behaviors: 탭하면 애니메이션 재생 등)을 실행할 수 있다. 우리 USDZ에는 아직 없다.
- 안드로이드 Chrome은 WebXR AR이 페이지 안에서 돌아서 AR 중에도 이 문서의 코드가 동작한다.
  WebXR이 안 되는 기기에서 model-viewer가 Scene Viewer(별도 앱)로 넘어가면 아이폰 AR과 같다.

## 7. 그 밖에

- **좌표**: 바닥이 y = 0, 개의 앞이 +z, 개의 왼쪽이 +x. 몸은 원점 근처에서만 움직인다.
  전환 중 최대 7cm 이동하고, `play_dead`·`rollover`는 제자리로 돌아온다. AR에서는 원점을 바닥 평면에 놓으면 된다.
- **숨쉬기**: 루프 클립 안에 들어 있다(`breathe_jnt` 스케일). 따로 할 것 없음.
- **속도**: `behavior.playback_rate`(지금은 1.0)를 `mixer.timeScale`에 넣는다.
- **라이선스 (꼭)**: `play_dead`와 전환 4개(`sit_down`, `stand_up`, `lie_down`, `get_up`)는
  AI4Animation SIGGRAPH 2018 개 모션캡처(Zhang, Starke et al., "Mode-Adaptive Neural Networks for Quadruped Motion Control")를 썼다.
  이 데이터는 **CC BY-NC 4.0**이라 앱에 출처를 표시해야 하고 상업적으로 쓸 수 없다.
  출처 예: `Dog motion capture: AI4Animation (Zhang, Starke et al., SIGGRAPH 2018), CC BY-NC 4.0`.
  `shake`는 Pexels 영상(출처 표시 불필요)을 MoCapAnything(MIT)으로 읽은 것이다.
- 예전 GLB에 `lie_flop`, `lie_square` 클립이 있으면 무시한다(11-06에 없앰).
