"""RH-P12-RN 그리퍼 턱 기하 (2026-10-07 측정). 파지 자세를 수치로 지정할 때 쓴다.

왜 필요한가
-----------
`core/kimm_door_traj.py` 의 `handle_hand` 는 **손끝 기준점**(numpy FK 체인의 끝 = USD 의
`arm_r_link7` 에서 로컬 −z 로 0.2 m 간 점)의 위치·자세만 정한다. 그런데 실제로 막대를 물는 것은
손가락 패드이고, 패드는 그리퍼 열림각 θ(`gripper_r_joint1`)에 따라 움직인다.
"막대가 패드 어디에 물렸는가"를 지정하거나 판정하려면 아래 관계식이 필요하다.

측정 근거
---------
URDF `assets/ffw_sg2_follower.urdf` 의 그리퍼 관절:

    gripper_r_joint1  base -> r1   xyz (0, +0.008, 0.048)  axis +x   limit 0~1.1
    gripper_r_joint2  r1   -> r2   xyz (0, +0.0493634, 0.0285) axis −x  mimic joint1 (limit 0~1.0)
    gripper_r_joint3/4 왼쪽 거울상 (joint3 limit 0~1.1, joint4 limit 0~1.0)

joint2/4 가 joint1 을 반대로 따라가므로 θ ≤ 1.0 에서 두 패드는 서로 **평행**하고, θ > 1.0 에서는
joint2/4 가 한계(1.0)에 걸려 패드가 가위처럼 기운다. 그래서 턱 간격이 좁아질수록 끝이 벌어진다.

패드 콜라이더는 손가락 메시의 **convexHull** 이다 (`collisions/*/node_STL_BINARY_`,
`physics:approximation = convexHull`). 링크 로컬 치수(측정):

    길이(z)  −0.015 ~ +0.045 m      폭(x) ±0.013 m
    안쪽면(y) −0.0039 m 에서 평면    두께(y) 밑동 0.022 m → 끝 0.008 m 로 가늘어짐

시뮬레이터 대비 모델 보정(`CAL_AXIS`, `CAL_JAW`)은 같은 관절값에서 Isaac 의 실제 패드 중앙과
이 식의 예측을 비교해 얻었다 (네 자세 평균, 잔차 1 mm 이내).
"""
import math

PAD_INNER = 0.0039      # 패드 원점 → 안쪽면 (집는 축 방향)
PAD_LEN = 0.045         # 패드 원점 → 끝 (손가락 길이 방향)
PAD_HALF_WIDTH = 0.013  # 패드 폭 절반 (막대 축 방향)
PAD_THICK_BASE = 0.022  # 밑동 두께 (안쪽면에서 등까지)
PAD_THICK_TIP = 0.008   # 끝 두께
MIMIC_LIMIT = 1.0       # joint2/4 한계. θ 가 이보다 크면 패드가 기운다
MECH_LIMIT = 1.094      # 허공에서 완전히 닫히는 값

CAL_AXIS = 0.00354      # 실제 패드 중앙이 모델보다 손가락 방향으로 이만큼 앞
CAL_JAW = -0.00268      # 집는 축 방향 보정


def jaw_half(theta):
    """턱 중앙 → 한쪽 패드 **원점** 거리 (m). 안쪽면은 여기서 PAD_INNER 안쪽."""
    return 0.008 + 0.0493634 * math.cos(theta) - 0.0285 * math.sin(theta)


def jaw_gap(theta):
    """두 패드 안쪽면 사이 간격 (m)."""
    return 2.0 * (jaw_half(theta) - PAD_INNER)


def theta_for_gap(gap):
    """안쪽면 간격이 gap 이 되는 θ. 단조 감소라 이분 탐색."""
    lo, hi = 0.0, MECH_LIMIT
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if jaw_gap(mid) > gap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pad_axis_offset(theta):
    """손끝 기준점 → 패드 원점 (손가락 방향 a 성분, m). 음수 = 패드가 기준점보다 손목 쪽."""
    return 0.078 + 0.048 + 0.0493634 * math.sin(theta) + 0.0285 * math.cos(theta) - 0.2


def pad_thickness(depth):
    """패드 원점에서 depth 만큼 간 지점의 두께 (m). 밑동 0.022 → 끝 0.008 선형."""
    t = min(max(depth / PAD_LEN, 0.0), 1.0)
    return PAD_THICK_BASE + (PAD_THICK_TIP - PAD_THICK_BASE) * t


def hand_target(bar, axis, jaw, theta, depth, calibrate=True):
    """막대 중심이 '패드 원점에서 depth 만큼 간 자리'에 오도록 하는 손끝 기준점 위치.

    bar  : 막대 중심 월드 좌표
    axis : 손가락이 가리키는 방향 a (단위벡터)
    jaw  : 집는 축 (단위벡터, a 와 수직)
    """
    mid = bar - depth * axis
    if calibrate:
        mid = mid + CAL_AXIS * axis + CAL_JAW * jaw
    return mid - pad_axis_offset(theta) * axis


def bar_span(jaw, width=0.011, height=0.012):
    """막대(단면 width×height, y 축으로 긴 봉)를 집는 축 방향으로 재면 몇 mm 인가.

    집는 축이 문 법선과 θ 만큼 기울면 비스듬히 물게 되므로 유효 두께가 커진다.
    """
    return abs(jaw[0]) * width + abs(jaw[2]) * height
