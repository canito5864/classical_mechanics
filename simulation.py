import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Streamlit 기본 설정
st.title("운동 시뮬레이션: 1차원, 2차원, 3차원 운동")
st.sidebar.title("운동 종류 및 설정")

# 사용자로부터 운동 종류 선택
motion_type = st.sidebar.selectbox(
    "시뮬레이션할 운동을 선택하세요:",
    ["1차원 운동", "2차원 운동", "3차원 운동"]
)

g = 9.81  # 중력 가속도 (m/s²)
rho = 1.225  # 공기 밀도 (kg/m³)

# --- 1차원 운동 ---
if motion_type == "1차원 운동":
    st.header("1차원 운동 시뮬레이션")
    # 사용자 입력
    initial_velocity = st.sidebar.number_input("초기 속도 (m/s)", value=10.0)
    acceleration = st.sidebar.number_input("가속도 (m/s²)", value=-9.81)
    drag_coefficient = st.sidebar.number_input("공기 저항 계수 (Cd)", value=0.0)
    mass = st.sidebar.number_input("질량 (kg)", value=1.0)
    time_duration = st.sidebar.number_input("운동 시간 (s)", value=5.0)

    # 계산
    dt = 0.01
    t = np.arange(0, time_duration, dt)
    v = [initial_velocity]
    x = [0]

    for i in range(1, len(t)):
        drag_force = drag_coefficient * v[-1]**2 / mass
        a = acceleration - (drag_force if v[-1] > 0 else -drag_force)
        v.append(v[-1] + a * dt)
        x.append(x[-1] + v[-1] * dt)

    # 결과 시각화
    fig, ax = plt.subplots()
    ax.plot(t, x, label="위치 (x)")
    ax.plot(t, v, label="속도 (v)", linestyle='--')
    ax.set_title("1차원 운동")
    ax.set_xlabel("시간 (s)")
    ax.set_ylabel("값")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

# --- 2차원 운동 ---
elif motion_type == "2차원 운동":
    st.header("2차원 운동: 포물선 운동 (공기 저항 포함)")
    # 사용자 입력
    initial_velocity = st.sidebar.number_input("초기 속도 (m/s)", value=50.0)
    angle = st.sidebar.number_input("발사 각도 (°)", value=45.0)
    drag_coefficient = st.sidebar.number_input("항력 계수 (Cd)", value=0.47)
    area = st.sidebar.number_input("단면적 (m²)", value=0.05)
    mass = st.sidebar.number_input("질량 (kg)", value=1.0)

    # 초기 값 설정
    angle_rad = np.radians(angle)
    v_x = initial_velocity * np.cos(angle_rad)
    v_y = initial_velocity * np.sin(angle_rad)

    # 수치 계산
    dt = 0.01
    t = [0]
    x = [0]
    y = [0]
    vx = [v_x]
    vy = [v_y]

    while y[-1] >= 0:  # 공이 땅에 닿을 때까지 반복
        v = np.sqrt(vx[-1]**2 + vy[-1]**2)
        drag_force = 0.5 * drag_coefficient * rho * area * v**2
        ax = -drag_force * vx[-1] / (mass * v)
        ay = -g - drag_force * vy[-1] / (mass * v)
        vx.append(vx[-1] + ax * dt)
        vy.append(vy[-1] + ay * dt)
        x.append(x[-1] + vx[-1] * dt)
        y.append(y[-1] + vy[-1] * dt)
        t.append(t[-1] + dt)

    # 결과 시각화
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("포물선 운동 궤적")
    ax.set_xlabel("수평 거리 (m)")
    ax.set_ylabel("수직 거리 (m)")
    ax.grid()
    st.pyplot(fig)

# --- 3차원 운동 ---
elif motion_type == "3차원 운동":
    st.header("3차원 운동: 행성 궤도")
    # 사용자 입력
    semi_major_axis = st.sidebar.number_input("장축 반경 (AU)", value=1.0)
    eccentricity = st.sidebar.number_input("이심률", value=0.1)
    inclination = st.sidebar.number_input("궤도 경사각 (°)", value=0.0)

    # 궤도 계산
    num_points = 500
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = r * np.sin(np.radians(inclination)) * np.sin(2 * theta)

    # 3D 그래프 출력
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_title("행성 궤도 시뮬레이션")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")
    st.pyplot(fig)
