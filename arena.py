import argparse 
from collections import deque 
import time 
import jax 
from jax import vmap 
import mujoco 
from mujoco import mjx 
from mujoco.mjx import Data

import sys 
from jax import numpy as jp 
import threading 
from queue import Queue

class BattleArena: 
    def __init__(self) -> None:

        self.model = mujoco.MjModel.from_xml_path("models/arena.xml")
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        self.model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        self.model.opt.iterations = 4 
        self.model.opt.ls_iterations = 4
            
        self.q_function = None
        self.q_params = None

        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)
            
        self.frame_skip = 5
            
        # --- ПАРАМЕТРЫ РОБОТА И ЦЕЛИ ИЗ PYBULLET ---
        self.agent_count = 1  
        obs_size = (self.mjx_model.nq - 2) + self.mjx_model.nv + self.mjx_model.nv
        self.observation_space_shape = (obs_size,) 
        self.action_space_shape = (self.mjx_model.nu,)

        self.ctrlrange_high = 1.0
        self.ctrlrange_low = -1.0
        self.max_steps = 1000

        # Целевая позиция из PyBullet окружения
        self.target_position = jp.array([5.0, 0.0, 0.10])
        # Начальное расстояние до цели (для нормирования ухода в сторону)
        self.initial_dist = jp.sqrt(self.target_position[0]**2 + self.target_position[1]**2)

        # Вычисляем масштаб действий под радианы
        self.action_scale = jp.array([jp.deg2rad(45.0), jp.deg2rad(45.0), jp.deg2rad(70.0)] * 6)

        self.elbow_site_names = ["elbow_one", "elbow_two", "elbow_three", "elbow_four", "elbow_five", "elbow_six"]
        self.elbow_site_ids = jp.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.elbow_site_names
        ])
    
    def reset(self, rng: jax.Array):
        data = mjx.make_data(self.mjx_model)
        rng_pos, rng_vel, rng_next = jax.random.split(rng, 3)

        qpos = data.qpos
        # Ставим робота в начальную точку на высоту 0.2 (как INIT_POSITION в PyBullet)
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, 0.15]))
            
        # Инициализация моторов в диапазоне от -10 до +10 градусов (pi / 18)
        num_joints = self.mjx_model.nu
        joint_noise = jax.random.uniform(rng_pos, (num_joints,), minval=-jp.pi/18.0, maxval=jp.pi/18.0)
        qpos = qpos.at[7:7+num_joints].set(joint_noise)

        # Скорости
        low, hi = -0.01, 0.01
        qvel = jax.random.uniform(rng_vel, (self.mjx_model.nv,), minval=low, maxval=hi)
            
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
            
        obs = self.get_obs(data)
        com_pos = data.subtree_com[1]
            
        return dict(
            data=data,
            step=0,
            obs=obs,
            rng=rng_next, 
            last_com=com_pos, 
        )
    
    def get_obs(self, data: Data):
        obs = jp.concatenate((
            data.qpos[2:],             
            data.qvel,           
            data.qfrc_actuator   
        ))
        return obs

    def compute_reward(self, old_data, new_data):
        dt = self.model.opt.timestep * self.frame_skip
        curr_com = new_data.subtree_com[1] # Текущая позиция корпуса [x, y, z]

        # Веса из PyBullet
        w1 = 12.0  # Вес за приближение к цели
        w2 = 0.4   # Вес за удержание высоты цели
        w3 = 0.0   # Вес за энергозатраты
        w4 = 0.4   # Вес за раскачивание корпуса (Roll/Pitch)
        w5 = 15.0   # Вес штрафа за уход вбок по Y (НОВЫЙ)
        w6 = 150.0 # Вес штрафа за касание локтями земли (настраивайте вручную)
        w7 = 5.0  # Вес мгновенного (барьерного) штрафа за факт касания (НОВЫЙ)
        w8 = 5.0 # Вес штрафа за поворот (можно настроить)
            

        old_com = old_data.subtree_com[1]
        velocity_x = (curr_com[0] - old_com[0]) / dt
        target_reward = w1 * velocity_x

        # Штраф за движение по Y
        y_position = curr_com[1] # Текущее положение робота на оси Y
        sideways_penalty = w5 * (y_position ** 2) 


        # 2. ШТРАФ ЗА ВЫСОТУ
        height_penalty = w2 * jp.abs(curr_com[2] - self.target_position[2])

        # 3. ПОТРЕБЛЕНИЕ ЭНЕРГИИ (активные суставы находятся начиная с индекса 6 в qvel/qfrc)
        speeds = new_data.qvel[6:]
        torques = new_data.qfrc_actuator[6:]
        consumption = dt * jp.abs(jp.sum(speeds * torques))
        power_penalty = w3 * consumption

        # 4. РАСКАЧИВАНИЕ КОРПУСА (Конвертируем кватернион MJX в углы Roll/Pitch)
        qw, qx, qy, qz = new_data.qpos[3], new_data.qpos[4], new_data.qpos[5], new_data.qpos[6]
            
        # Roll (угол крена)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = jp.arctan2(sinr_cosp, cosr_cosp)
            
        # Pitch (угол тангажа)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = jp.clip(sinp, -0.999999, 0.999999) # избегаем NaN в arcsin
        pitch = jp.arcsin(sinp)
            
        shaking_penalty = w4 * (jp.abs(roll) + jp.abs(pitch))
            
        # Барьерный штраф (НОВЫЙ): если хотя бы один локоть ниже порога, штрафуем сразу на w7
        # 5. ШТРАФ ЗА КАСАНИЕ ЛОКТЯМИ ЗЕМЛИ (СТУПЕНЧАТЫЙ)
        elbow_heights = new_data.site_xpos[self.elbow_site_ids, 2]
        elbow_threshold = 0.028  # Порог высоты локтя (2.5 см)
            
        # Получаем булев массив (True там, где локоть ниже порога, False - где выше)
        is_touching = (elbow_heights < elbow_threshold)
            
        # Суммируем количество коснувшихся ног (True превращается в 1, False в 0)
        num_touching_legs = jp.sum(is_touching)
            
        # w7 — фиксированный штраф за ОДНУ коснувшуюся ногу. 
        elbow_penalty = w7 * num_touching_legs

        qw, qx, qy, qz = new_data.qpos[3], new_data.qpos[4], new_data.qpos[5], new_data.qpos[6]
            
        # Преобразование кватерниона в углы Эйлера (XYZ, как в MuJoCo)
        # yaw (рысканье)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = jp.arctan2(siny_cosp, cosy_cosp)
            
        yaw_penalty = w8 * jp.abs(yaw)

        # Итоговая награда
        reward = target_reward - height_penalty - power_penalty - shaking_penalty - sideways_penalty - elbow_penalty - yaw_penalty
        #reward *= 0.01

        return reward


    def validation_reward(self, old_com, new_com):
        # Метрика прогресса к цели (на сколько метров продвинулся к точке)
        old_dist = jp.sqrt((self.target_position[0] - old_com[0])**2 + (old_com[1] - self.target_position[1])**2)
        new_dist = jp.sqrt((self.target_position[0] - new_com[0])**2 + (new_com[1] - self.target_position[1])**2)
            
        return old_dist - new_dist


    def step(self, state0, control):
        scaled_control = control * self.action_scale

        # Интегрирование физики
        data, _ = jax.lax.scan(
            lambda d0, _: (mjx.step(self.mjx_model, d0.replace(ctrl=scaled_control)), None), 
            state0['data'], (), self.frame_skip
        )

        new_step = state0['step'] + 1
        steps_limit_reached = (new_step >= self.max_steps)
        spider_com = data.subtree_com[1]

        # Расчет награды
        reward = self.compute_reward(state0['data'], data)

        # Условия окончания из PyBullet (roll/pitch вне диапазона +- 30 градусов)
        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            
        # Вычисление Roll/Pitch
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = jp.arctan2(sinr_cosp, cosr_cosp)
            
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = jp.clip(sinp, -0.999999, 0.999999)
        pitch = jp.arcsin(sinp)
            
        # Булевы условия конца эпизода
        limit_angle = jp.deg2rad(45.0) 
        pitch_exceeded = (pitch > limit_angle) | (pitch < -limit_angle)
        roll_exceeded = (roll > limit_angle) | (roll < -limit_angle)

        fell_or_collapsed = (spider_com[2] < 0.03)
            
            
        # Слишком далеко от цели (расстояние выросло более чем в 2 раза от изначального)
        curr_dist = jp.sqrt((self.target_position[0] - spider_com[0])**2 + (spider_com[1] - self.target_position[1])**2)
        too_far = (curr_dist > 2.0 * self.initial_dist)

        done = jp.where(steps_limit_reached | fell_or_collapsed | pitch_exceeded | roll_exceeded | too_far, 1.0, 0.0)

        # Метрика прогресса для валидации
        val_reward = self.validation_reward(state0['last_com'], spider_com)

        observations = self.get_obs(data)
        new_state = dict(
            data=data, 
            step=new_step, 
            obs=observations, 
            rng=state0['rng'], 
            last_com=spider_com
        ) 

        next_state = jax.lax.cond(done > 0.5, self.reset, lambda _ : new_state, new_state['rng'])

        return (
            next_state, 
            observations, 
            jp.array([reward]).reshape(-1), 
            jp.array([done]).reshape(-1), 
            val_reward
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Spider Rescue Mission Simulation')
    
    parser.add_argument('--agent', type=str, default=None,
                        help='Path to agent .flax file (default: agent.flax in root)')
    
    parser.add_argument('--norm', type=str, default='obs-norm/obs_norm_last.npz',
                        help='Path to normalization values (.npz)')
    
    parser.add_argument('--save-mode', type=str, choices=['none', 'images', 'positions', 'both'],
                        default='none', help='What data to save')
    
    args = parser.parse_args()
    
    if args.agent is None:
        if os.path.exists('agent.flax'):
            args.agent = 'agent.flax'
        else:
            print("Warning: No agent file found, will use random actions or wait for input.")
    
    return args


def get_agent_display_name(agent_path):
    dirname = os.path.dirname(agent_path)
    if dirname == '':
        return f"./{os.path.basename(agent_path)}"
    return dirname


def finish(image_queue, image_saver_thread):
    if image_queue is not None:
        image_queue.put((None, None, None, None))
        print("Waiting frames to be saved as images...")
        image_saver_thread.join()
    sys.exit(0)


def save_images(queue):
    while True:
        frame, pixels, xpos, xmat = queue.get()
        if frame is None:
            break
        if xpos is not None and xmat is not None:
            np.savez(f"outputs/pos-data/{frame:05d}.npz", geom_xpos=xpos, geom_xmat=xmat)
        if pixels is not None:
            plt.imsave(f"outputs/images/{frame:05d}.png", pixels)
        
        
@jax.jit            
def normalize_obs(obs, obs_rms_mean, obs_rms_var):
    return jax.numpy.clip((obs - obs_rms_mean) / jax.numpy.sqrt(obs_rms_var + 1e-8), -10.0, 10.0)
            
            
def main():
    args = parse_args()

    print("Starting Spider Rescue Mission with parameters:")
    print(f"Agent: {args.agent}")
    print(f"Normalization: {args.norm}")
    print(f"Save mode: {args.save_mode}")

    ba = BattleArena()

    batch_size = 64
    key = jax.random.split(jax.random.key(42), batch_size)
    batched_state = vmap(ba.reset)(key)
    obs_shape = batched_state['obs'].shape[-1] 
    print('Batch observation shape:', batched_state['obs'].shape) 

    agent = ActorSimple_skip(
        ba.action_space_shape[0], 
        ba.ctrlrange_high, ba.ctrlrange_low,
        512, 512
    ) 

    from flax import serialization
    init_key = jax.random.key(0)
    dummy_obs = batched_state['obs'][0:1] 

    if args.agent and os.path.exists(args.agent):
        with open(args.agent, 'rb') as f:
            agent_bytes = f.read()
        agent_params = serialization.from_bytes(
            agent.init(init_key, dummy_obs)['params'], 
            agent_bytes
        )
        agent_params = jax.device_put(agent_params)
        print("Agent loaded from file.")
    else:
        agent_params = agent.init(init_key, dummy_obs)['params']
        print("Using RANDOM weights.")

    obs_shape = batched_state['obs'].shape[-1]

    obs_rms_mean = jp.zeros(obs_shape)
    obs_rms_var = jp.ones(obs_shape)
    if os.path.exists(args.norm):
        obs_norm_values = np.load(args.norm)
        if len(obs_norm_values['obs_mean']) == obs_shape:
            obs_rms_mean = jax.numpy.array(obs_norm_values['obs_mean'])
            obs_rms_var = jax.numpy.array(obs_norm_values['obs_var'])
            print("-> Normalization loaded.")
        else:
            print(f"-> Norm shape mismatch! Expected {obs_shape}. Using default.")
    else:
        print("-> Using default normalization.")

    image_queue = None
    image_saver_thread = None
    if args.save_mode != 'none':
        os.makedirs('outputs', exist_ok=True)
        os.makedirs(os.path.join('outputs', 'images'), exist_ok=True)
        os.makedirs(os.path.join('outputs', 'pos-data'), exist_ok=True)
        image_queue = Queue()
        image_saver_thread = threading.Thread(target=save_images, args=(image_queue,))
        image_saver_thread.start()

    inference_key, env_key = jax.random.split(jax.random.key(42), 2)
    
    jit_reset = jax.jit(ba.reset)
    jit_step = jax.jit(ba.step)
    
    jit_get_action = jax.jit(agent.get_action)

    state = jit_reset(env_key)

    width, height = 1280, 720
    ba.model.vis.global_.offwidth = width
    ba.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(ba.model, width=width, height=height)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1 
    camera.distance = 2.5 

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Spider Rescue: {args.agent}")
    clock = pygame.time.Clock()
    keymap = {pygame.K_ESCAPE: 0.0, pygame.K_SPACE: 0.0}

    rewards_history = deque([0.0]*1000, maxlen=1000)
    frame = 0
    # =======================================================
    # БЛОК ВЫГРУЗКИ ДАННЫХ ДЛЯ ТЕСТИРОВАНИЯ В EXCEL
    # =======================================================
    print("\n" + "="*40)
    print("ВЫГРУЗКА ДАННЫХ ДЛЯ EXCEL (МОТОР 1)")
    
    test_obs = jp.expand_dims(state['obs'], 0)
    test_obs_norm = normalize_obs(test_obs, obs_rms_mean, obs_rms_var)
    test_key = jax.random.key(777) 
    
    mean, log_std = agent.apply({'params': agent_params}, test_obs_norm)
    
    std = jp.exp(log_std)
    noise = jax.random.normal(test_key, shape=mean.shape)
    x_t = mean + std * noise
    
    action_scale = (ba.ctrlrange_high - ba.ctrlrange_low) / 2.0
    
    normal_log_prob = -0.5 * (((x_t - mean) / std) ** 2 + 2 * log_std + jp.log(2 * jp.pi))
    log_tanh_deriv = 2.0 * (jp.log(2.0) - x_t - jax.nn.softplus(-2.0 * x_t))
    single_log_prob = normal_log_prob - (jp.log(action_scale) + log_tanh_deriv)
    
    print(f"Mean      (M): {float(mean[0, 0]):.6f}")
    print(f"Log_Std   (L): {float(log_std[0, 0]):.6f}")
    print(f"Noise     (E): {float(noise[0, 0]):.6f}")
    print(f"Итог Log_Prob: {float(single_log_prob[0, 0]):.6f}")
    print("="*40 + "\n")
    # =======================================================

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                finish(image_queue, image_saver_thread)
            if event.type == pygame.KEYDOWN:
                keymap[event.key] = 1.0
            if event.type == pygame.KEYUP:
                keymap[event.key] = 0.0

        if keymap[pygame.K_ESCAPE] > 0.0:
            finish(image_queue, image_saver_thread)

        if keymap[pygame.K_SPACE] > 0.0:
            state = jit_reset(jax.random.key(time.time_ns()))

        mjx.get_data_into(ba.data, ba.model, state['data'])
        mujoco.mj_forward(ba.model, ba.data)
        renderer.update_scene(ba.data, camera=camera)
        pixels = renderer.render()

        if image_queue is not None:
            px = pixels if args.save_mode in ['images', 'both'] else None
            xp = ba.data.xpos if args.save_mode in ['positions', 'both'] else None
            xm = ba.data.xmat if args.save_mode in ['positions', 'both'] else None
            image_queue.put((frame, px, xp, xm))

        pixels_surf = np.swapaxes(pixels, 0, 1)
        screen.blit(pygame.surfarray.make_surface(pixels_surf), (0, 0))

        scale = 40
        cx, cy = 1080, 600
        spider_pos = state['last_com']

        pygame.draw.rect(screen, (0, 255, 0), (cx-100, cy-100, 200, 200), 2)
        pygame.draw.circle(screen, (255, 0, 0), (int(cx + spider_pos[0]*scale), int(cy - spider_pos[1]*scale)), 8)

        inf_key, inference_key = jax.random.split(inference_key)

        obs_batched = jp.expand_dims(state['obs'], 0)
        
        obs_norm = normalize_obs(obs_batched, obs_rms_mean, obs_rms_var)

        _, _, action = jit_get_action(agent_params, obs_norm, inf_key)

        state, _, reward, done, val_dist = jit_step(state, jp.squeeze(action))

        pygame.display.flip()
        
        reward_val = float(reward[0])
        rewards_history.append(reward_val)

        if frame % 20 == 0:
            print(f"Step: {state['step']} | Rew: {reward_val:.4f} | Dist: {val_dist:.2f} | FPS: {clock.get_fps():.1f}")
    
        clock.tick(50) 
        frame += 1

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from flax import serialization
    import pygame
    from pygame.locals import QUIT
    from flax import serialization
    from agent import ActorSimple_skip
    import numpy as np
    import argparse
    import os
    import sys
    import time
    import threading
    from queue import Queue
    from collections import deque
    main()