# kimm_door v2 — 문 D8 열기 (재작성)

`~/world model develop/kimm_door_scripts/` (v1) 에서 **검증된 것만 가져오고 나머지는 다시 짠다.**

v1 기록: `~/WORKLOG_2026-10-03.md`

---

## 왜 다시 짜는가

v1 은 세 스크립트 모두 설계대로 돌아가지만 문이 안 열린다. 막히는 구간은 **손잡이 파지**다.
2026-10-07 측정으로 원인 네 가지를 찾아 고쳤고(아래 "원인 규명"), 지금은 파지가 성립한다.
남은 문제는 당기는 중 막대가 패드 끝으로 밀려 빠지는 것이다 ("남은 문제" 절).
**손목 강성은 원인이 아니었다** — 베이스가 굴러 손이 40 mm 못 미친 것이 가장 큰 원인이었다.

v1 구조의 문제는 따로 있다.

| 문제 | v2 에서 |
|---|---|
| 상수가 코드 여기저기 흩어짐 (좌표, DOF 인덱스, 게인) | `config/d8.yaml` 한곳 |
| `kimm_door_motion.py` 가 구간·시간축·실행을 다 들고 있음 | `motion/` 과 `tools/` 분리 |
| 환경 접속 코드가 실행기에 섞임 | `core/` 로 분리 |
| 액추에이터 게인을 건드릴 방법이 없음 | 설정에서 주입 |

---

## 폴더

```
config/d8.yaml      좌표·DOF·게인·판정값. 출처 주석 필수
core/               v1 에서 가져온 검증된 기하·IK (수정 금지 대상)
scenes/             실행용 루트 레이어
assets/             URDF (core/urdf_ik_scan.py 가 ../assets/ 로 찾는다)
motion/gripper_geom.py  그리퍼 턱 기하 (턱 간격↔θ, 패드 치수·보정). 2026-10-07 측정
tools/run.py        Isaac Sim 단독 실행기 (몸통·바퀴 고정, 왼쪽 손가락 드라이브 복사)
tools/grasp_probe.py  파지 진단 — 막대가 패드 어디에 물렸고 어디로 밀리는지 측정
logs/               실행 로그
```

### `core/` — v1 에서 그대로 가져온 것 (검증됨)

| 파일 | 왜 가져왔나 |
|---|---|
| `kimm_door_traj.py` | 기하·IK. 생성된 관절값이 성공 에피소드와 **0.007 rad 이내 일치**. IK 오차 0.5~0.6mm |
| `urdf_ik_scan.py` | FK/IK 체인. numpy 전용이라 Isaac 없이도 돈다 |
| `kimm_traj.py` | `ik_w`, `w2r` 좌표 변환 |
| `door_pose_eval.py` | `ik_post`, `Q_NOM_R`. 비용 함수에 `q[3] > -0.2` 벌점 10 이 있다 |
| `lift_scan.py`, `door_free_check.py` | 조사용 |

**이 파일들은 다시 짜지 않는다.** v1 에서 검증된 유일한 부분이다.

### `scenes/kimm_door_d8_run.usda`

```
subLayers = [ fix_gripper_mimic.usda,          # 그리퍼 4관절 mimic 보정 (없으면 옆 파지 낙하)
              kimm_room_static.usda,            # D8 박스 충돌체 (없으면 손잡이가 아예 없음)
              env2_kimm_room_ver1.usd ]         # 원본 씬
upAxis = "Z"; metersPerUnit = 1; defaultPrim = "ffw_sg2_follower"
```

**`kimm_room_static.usda` 를 v2 폴더로 옮기지 말 것.** payload 가 상대경로(`./LAB_3/`)라
`LAB_3/` 가 있는 `kroc2027/world/` 에 있어야만 방 지오메트리가 로드된다.

**스테이지 메타데이터(`upAxis` 등)는 subLayer 에서 합성되지 않는다.** 루트에 다시 선언하지 않으면
Y-up 으로 해석돼 방이 90도 눕는다.

---

## 알려진 사실 (v1 에서 실측, 재확인 불필요)

### 환경

```
DOF 31개   arm_r_joint1~7 -> [6, 11, 14, 16, 18, 20, 22]
           gripper_r_joint1 -> 25,  lift_joint -> 1
D8         collisionEnabled=False, angularDamping=5.0
           자식 콜라이더: col_panel, col_bar, col_leg_a, col_leg_b
드라이브   USD 각도 드라이브는 **도 단위**다 (rad 로 보려면 ×57.2958).
           arm_r_joint1  stiffness 1730.57 / damping 0.692 / maxForce 1000
           arm_r_joint7  stiffness 24.21 (= 1387 N·m/rad) / damping 0.00968
           gripper_r_joint1  stiffness 0.497 (= 28.5 N·m/rad) / damping 0.000199 / limit [0, 63.03도]
           gripper_r_joint2~4  드라이브 없음 — mimic 추종자 (gearing -1). joint3 는 tools/run.py 가 드라이브를 복사한다
바퀴       left/right/rear_wheel_drive  maxForce **1.0 N·m** → 팔을 뻗으면 베이스가 굴러 밀린다
           lock_wheels 로 limit 0/0 잠금 (필수)
리프트     잠금 전 위치 드라이브만 사용했을 때 ~1.4cm 처짐 (-0.2636 vs 목표 -0.25)
```

### 문 구조 (x 좌표)

```
문짝 표면 col_panel   x [-2.227, -2.212]   z [0.094, 0.851]
다리 col_leg_a/b      x [-2.252, -2.227]   y -3.218~-3.205 / -3.070~-3.056 (막대 양끝)
손잡이 막대 col_bar   x [-2.261, -2.250]   y [-3.207, -3.068]   z [0.814, 0.826]
  -> 단면 11 x 12 mm, 길이 139 mm (y 축 봉)
캐비닛 앞면           x -2.2275 (z <= 0.8815),  상판은 x -2.2344 (z 0.8815~0.8946)
  -> 손잡이 뒤 틈 = 막대 뒤(-2.250)에서 캐비닛 앞면까지 **22.5 mm**
  -> 막대 중심에서 문짝까지 28 mm. 패드(길이 45 mm)가 이 안에 들어가면 문짝을 민다
```

### 판정 기준

이 씬(실제 손잡이 형상) 기준 — 2026-10-07 재측정:

| 신호 | 성공 | 실패 |
|---|---|---|
| 그리퍼 실제값 | **0.95~1.05** (막대에 막힘, 실측 0.958~0.977) | **1.085 이상** (허공, 한계 1.094) |
| 막대 접촉 | l1,l2,r1,r2 네 손가락 | 한쪽만 또는 0건 |
| 클램프 힘 | 25~47 N | 0 |
| 문 각도 | > 80도 | — |

성공 에피소드의 **0.576** 은 원본 convexHull 문(두께 49 mm 덩어리)을 물었을 때의 값이라
이 씬에는 적용할 수 없다. 같은 이유로 "접촉 시 joint7 편차 +0.553 rad" 도 그 씬의 현상이다.

---

## 원인 규명 — 2026-10-07

이전 기록은 파지 실패의 원인을 **손목 강성**으로 지목했다. 다시 측정해 보니 그게 아니었다.
세 가지 별개의 원인이 겹쳐 있었고, 셋 다 측정으로 확인하고 수정했다.

### 1. 베이스가 굴러 밀린다 (가장 큰 원인)

로봇은 모바일 베이스 위에 있고, 바퀴 구동 관절의 drive `maxForce` 가 **1 N·m** 뿐이다.
팔을 손잡이로 뻗는 동안 `left/right/rear_wheel_drive` 가 **0.43 rad** 돌아 로봇이 뒤로 밀린다.
그만큼(약 **40 mm**) 손이 손잡이에 못 미치므로, 손가락이 막대 끝만 걸치거나 아예 닿지 않는다.

- 성공 에피소드의 `obs/joint_pos_target` 19열을 그대로 재생해도 **접촉이 0건**이었던 이유가 이것이다
  (손목 강성을 1730/170/17 로 바꿔도 결과가 바뀌지 않았던 것도 같은 이유).
- 같은 관절값을 `set_joint_positions` 로 순간 이동시켜 재면 모델과 4 mm 안에서 일치한다.
  즉 기구학·IK 는 처음부터 맞았고, **실행 중 베이스가 움직인 것**이 문제였다.
- 수정: `tools/run.py` 의 `lock_wheels` 가 물리 시작 전에 바퀴 6관절을 limit 0/0 으로 잠근다.
  `step()` 은 베이스가 2 mm 넘게 움직이면 중단한다. `--selftest` 에서 베이스 이동 0.005 mm.

### 2. 왼쪽 손가락에는 드라이브가 없다

USD 에는 `gripper_r_joint1`(오른쪽)에만 드라이브가 있고 `joint3`(왼쪽)은 mimic 스프링으로만 따라간다.
접촉 하중이 걸리면 왼쪽이 뒤처진다 — 실측 joint1 0.976 / **joint3 0.708** (0.27 rad 차이).
그래서 막대에 **오른쪽 손가락만** 닿았다 (`r1`,`r2` 만 접촉).

- 실제 RH-P12-RN 은 한 모터가 링크로 양 손가락을 함께 구동한다. 그래서 오른쪽 드라이브를 왼쪽에
  복사하고 같은 목표를 보낸다 (`drive_left_finger`, `config/d8.yaml` 의 `gripper.drive_left`).
- 결과: 턱 중앙 오차 11.4 mm → 1.3 mm, **네 손가락(l1,l2,r1,r2) 모두 막대에 접촉**.

### 3. 성공 기록의 그리퍼 값 0.576 은 이 씬에서 나올 수 없다

원본 씬의 D8 은 **문 전체가 하나의 convexHull**(문짝+손잡이가 두께 49 mm 덩어리)이었다.
성공 에피소드는 그 덩어리를 비스듬히 물었고, 그때 턱 개구가 약 60 mm → `gripper_r_joint1` 0.576 이다.
v2 씬은 손잡이를 실제 형상(막대 11×12 mm, 뒤 틈 22.5 mm)으로 바꿨으므로 같은 값이 나올 수 없다.

- 이 씬에서 막대를 물면 **0.95~1.05** 에서 멈춘다 (실측 0.958~0.977, 클램프 25~47 N).
  허공을 쥐면 1.085 이상(기계 한계 1.094). `config/d8.yaml` 의 `grasp_range` 를 이 값으로 바꿨다.
- 즉 기존 판정 기준(0.45~0.70)은 **다른 씬의 값**이었고, 맞는 파지도 실패로 찍었다.

### 4. 손이 계획대로 들어가면 손가락 끝이 문짝에 닿는다

바퀴를 잠그면 손이 제자리에 서는데, 그러면 플래너 기본값에서 막대가 패드 원점에서 **14 mm** 에 물린다.
패드 길이는 45 mm 이므로 남은 31 mm 가 틈(막대 중심에서 문짝까지 28 mm) 밖으로 나가 문짝을 밀고,
그 반력으로 왼쪽 mimic 이 막힌다. `gripper.grasp_back = 0.008` 로 8 mm 뒤로 물리면 22 mm 가 되고
문짝까지 6 mm 여유가 남는다. `gripper.grasp_dz = 0.0027` 은 턱 중앙 보정이다
(모델이 예측한 턱 중앙보다 실제가 2.7 mm 아래 — `motion/gripper_geom.py` 참고).

---

## 남은 문제 — 당기는 중 막대가 패드 끝으로 밀려 빠진다

위 네 가지를 고치면 **파지는 성립한다** (네 손가락 접촉, grip 0.958~0.977, 클램프 25~47 N).
그런데 당기기 구간에서 막대가 패드 길이 방향으로 밀려 끝(45 mm)을 넘으면 턱이 벌어져 놓친다.

`tools/grasp_probe.py` 실측 (2026-10-07, 바퀴 잠금 + 왼쪽 드라이브 + 솔버 64/16):

| 당기기 스텝 | 문 각도 | 막대 깊이 | θ | 접촉 |
|---|---|---|---|---|
| 0 | 0.2° | 22 mm | 0.96 | 2점 30 N |
| 42 | 2.45° | 31.8 mm | 0.991 | 2점 27 N |
| 48 | 2.81° | 35.2 mm | 1.001 | 2점 25 N |
| 54 | 0.17° | **58.0 mm (이탈)** | 1.094 | 0점 |

손이 1 mm 갈 때 막대가 약 0.5 mm 밀린다. 즉 문이 하는 일은 손 이동량의 40% 정도다.

측정으로 **배제한** 원인:

- 마찰 부족이 아니다. 손가락·문에 마찰 1.0 머티리얼을 물려도 결과가 같다 (`friction.pad`).
- 문이 무거워서가 아니다. 문 14.3 kg, 경첩에 드라이브·마찰 없음, `angularDamping` 5.0 뿐.
  이 구간 각속도(0.07 rad/s)를 유지하는 데 필요한 힘은 손잡이에서 **약 1 N** 이다
  (각속도 0.5 rad/s 를 주면 1초에 61° 돈다). 클램프 25~47 N 에 비해 훨씬 작다.
- 그리퍼 힘 부족이 아니다. USD 드라이브 강성 0.497 은 **각도 단위**라 28.5 N·m/rad 다
  (이전 세션이 액추에이터 게인을 0.497~10 으로 "올린" 것은 사실상 57배 낮춘 것이었다).

남은 후보는 **턱 형상**이다. `joint2/4` 의 한계가 1.0 이라 θ > 1.0 에서 패드가 가위처럼 기울고
끝으로 갈수록 턱이 벌어진다. 막대가 끝 쪽으로 밀릴수록 간격이 커져 더 쉽게 빠지는 되먹임이 생긴다.
11~12 mm 막대를 물려면 θ ≈ 1.02 가 필요해 이 영역을 피할 수 없다.

### 다음에 시도할 것 (측정 근거 있는 순서)

1. **형상 구속 파지**: 손가락을 아래로 기울여 한쪽 패드 끝을 손잡이 뒤 틈(22.5 mm)에 넣고
   막대를 앞뒤로 물면, 당기는 힘을 패드 면이 받아 마찰에 의존하지 않는다.
   닫힌 문에서도 IK 가 0.02 mm 로 풀린다(기울기 25~45°, 몸통 −0.25). 2026-10-07 시험에서는
   패드 콜라이더(convexHull)가 근사 모델보다 두꺼워 삽입 중 캐비닛에 닿았다 (103 N).
   패드 hull 을 정확히 쓰면(메시 정점 → 실제 hull) 여유를 다시 계산할 수 있다.
2. 당기기 구간에서 **막대 깊이를 되먹임**해 손을 틈 쪽으로 조금씩 더 밀어 넣기 (깊이 유지 제어).
3. 손목(`arm_r_joint7`) 강성은 USD 실측 24.21(각도 단위 = 1387 N·m/rad)이다. 이 값을 바꾸는 것은
   위 1~2 를 해 본 뒤에, 성공 에피소드의 손목 비틀림(접촉 시 +0.553 rad)을 재현하는지로만 판단한다.
---

## 2단계는 1단계에 의존한다

`push_*` 구간은 문이 이미 **15도 이상** 열려 있어야 IK 해가 나온다.
닫힌 상태(0.03도)에서 실행하면 `phi` 후보가 하한 15.0 으로 깎이고 `pre_ik` 오차가 **109.6mm** 가 된다
(열린 상태 가정 시 0.6mm). v1 실측.

따로 검증하려면 문을 20도 열어 놓고 시작해야 한다 (v1 의 `--door_start 20`,
Isaac Lab 쪽은 `cfg.events.reset_door.params["angle_deg"] = (20.0, 20.0)`).

---

## v1 에서 쓰지 말아야 할 것

- `kimm_record/act_door/*.hdf5` — 문 열기 시연이 아니다. `fake_teleop_pub.py` 의 사인파 스윕(R²=1.0000)
- 성공 에피소드의 `success` attr — 4개 전부 `'False'` 로 잘못 찍혀 있다. `obs/door_angle` 로 판정할 것

## 참고 데이터

```
~/world model develop/kimm_door_episodes/
├── door_lift-0.25_success2_20261002.hdf5   801 MB  demo_0/demo_1, 각 1136 step
└── door_lift-0.45_success2_20261002.hdf5   952 MB  demo_0/demo_1, 각 1500 step
```
`obs/joint_pos(_target)` 19열 = arm_l(0-6), gripper_l(7), arm_r(8-14), gripper_r(15),
head1(16), head2(17), lift(18). 4개 데모 모두 `obs/door_angle` 이 0 -> 90도 도달.

## 실행 환경

```bash
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python <script>     # Isaac Sim 6.1
python3 <script>                                                # core/ 는 numpy 전용
```

## 간단 실행기 — 몸통 시작부터 고정

v1의 팔 궤적·IK를 사용하는 `tools/run.py`로 실행합니다. 리프트는 첫 물리 스텝 전에
지정 높이에 놓고 관절 하한·상한을 같은 값으로 잠급니다. 기본 높이는 −0.25 m이며,
`--lift -0.45`도 지원합니다. 실행 중 높이 오차가 1 mm를 넘으면 중단합니다.

```bash
cd ~/"world model develop"/kimm_door_v2

# 몸통 고정만 검사 (팔 궤적 실행 없음)
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/run.py --headless --selftest

# 닫힌 D8 손잡이 잡기 → 당기기 → 팔 복귀 → 밀기
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/run.py

# GUI에서 검증: 문 각도·몸통 높이 표시, 결과 창 유지
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/run.py --keep-open

# 낮은 몸통 높이로 실행
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/run.py --lift -0.45

# 이미 20도 열린 문에서 밀기 단계만 확인
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/run.py --phase push --door-start 20
```

그리퍼와 팔을 함께 보간해 놓기·정지 구간을 유지합니다. 실제 문 각도가 15도 미만이면
밀기를 실행하지 않습니다. 문이 80도를 넘으면 종료 코드 0, 미달이면 1입니다.
`--selftest`의 종료 코드 0은 몸통 고정 검사 통과를 뜻합니다.
잡기 구간 뒤에는 손잡이 막대와 양쪽 손가락의 실제 접촉을 확인하고,
그리퍼 값이 성공 기록에 근거한 0.45~0.70 rad 범위인지 검사합니다. 조건을 만족하지
못하면 당기기를 시작하지 않고 잡기 위치에서 멈춥니다. GUI는 손잡이를 가까이 보여 줍니다.
GUI는 로봇과 D8을 보는 시점으로 시작하며 `D8 Door Validation` 창에서 진행 단계,
문 각도, 몸통 높이와 최대 오차, 그리퍼 상태를 표시합니다. `--keep-open`에서는
검증 후 물리를 일시정지하고 결과 창을 유지합니다. IK 계산 중에도 화면 조작이 가능합니다.
`--capture-ui /tmp/d8_result.png`로 시작 화면(`d8_result_start.png`)과 결과 화면을 저장할 수 있습니다.

### 파지 검증 상태 (2026-10-07)

`OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/run.py --headless` 실측:

```
[v2] 몸통 시작 높이 -0.25 m, 리프트 관절 잠금, 바퀴 6관절 잠금
[v2] 솔버 반복: 위치 64, 속도 16
[v2] 왼쪽 손가락 드라이브 복사: stiffness 0.497 / damping 0.000199 / maxForce 1000
[v2] grasp  문 0.23 deg  grip 0.964
[v2] 파지 검사: grip=0.964, 범위=[0.9, 1.06], 손잡이 접촉=l1, l2, r1, r2   ← 통과
[v2] pull   문 0.02 deg  grip 1.091                                      ← 당기는 중 빠짐
[v2] 실패: 문이 0.0 deg 라 밀기 중단 (최소 15.0 deg)
     몸통 최대 오차 0.022 mm, 베이스 최대 이동 1.041 mm
```

- **파지 검사 통과**는 2026-10-07 이전에는 한 번도 없었다 (이전에는 한쪽 손가락만 닿거나 허공).
- 당기기에서 문 각도는 최고 2.8~6.4도까지 올라가다 막대가 빠지면 0도로 되돌아간다.
- `--selftest` 는 −0.25 m, −0.45 m 모두 통과 (리프트 오차 ≤ 1 mm, 베이스 이동 0.005 mm).
- 2단계(밀기)는 문이 15도 이상이어야 시작하므로 아직 실행되지 않는다.

진단은 `tools/grasp_probe.py` 로 재현한다:

```bash
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/grasp_probe.py
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/grasp_probe.py --no-wheel-lock   # 베이스 밀림 재현
OMNI_KIT_ACCEPT_EULA=YES ~/env_isaacsim/bin/python tools/grasp_probe.py --no-left-drive   # 한쪽 파지 재현
```

### 설정에서 바꿀 수 있는 값 (2026-10-07 추가)

| 키 | 기본값 | 근거 |
|---|---|---|
| `robot.wheel_joints` | 바퀴 6관절 | drive maxForce 1 N·m → 잠그지 않으면 베이스가 40 mm 밀린다 |
| `gripper.drive_left` | true | 실제 그리퍼는 한 모터가 양 손가락 구동. 끄면 왼쪽이 0.27 rad 뒤처진다 |
| `gripper.grasp_back` | 0.008 | 0 이면 패드 끝이 문짝에 닿는다. 8 mm 빼면 막대가 패드 22 mm 에 물린다 |
| `gripper.grasp_dz` | 0.0027 | 턱 중앙 보정 (모델 대비 실제가 2.7 mm 아래) |
| `gripper.grasp_range` | [0.90, 1.06] | 이 씬에서 막대를 물면 0.95~1.05, 허공은 1.085 이상 |
| `gripper.pull_rep` | 6 | 당기기 점당 반복. 단 `resample` 이 경로 길이로 보간하므로 속도에는 거의 영향 없음 |
| `solver.*` | 64 / 16 | 기본값으로는 11 mm 막대 접촉이 부정확하다 |
| `friction.pad` | null | 1.0 으로 올려도 당기기 결과가 같았다 (미끄러짐은 마찰 문제가 아니다) |
