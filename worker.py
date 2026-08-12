import threading
from queue import Queue
import logging 

import jax
from jax import jit, vmap
from jax import numpy as jp
import time
from collections import deque
import numpy as np
import sys
from flax import serialization
import os
from running_mean_std_jax import RunningMeanStd

np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    level=logging.WARNING
)

class WorkerThread(threading.Thread):
    def __init__(self, config, env, rng_key, result_queue: Queue, agent_queue:Queue, agent_config, name='Worker Thread'):
        super(WorkerThread, self).__init__(name=name)
        self.daemon = True
        self.config = config
        self.agent_config = agent_config
        
        self.output_queue = result_queue
        self.agent_queue = agent_queue
        self.running = False
        self.device = config['worker_device']
        
        self.env_batch_size = config['worker_batch_size']
        self.validation_batch_size = 64 
        self.total_batch_size = self.env_batch_size + self.validation_batch_size

        self.random_steps_count = config['random_steps_count']

        self.validant_last_reward = None
        self.validant_best_reward = None
        
        self.validation_steps = 2000
        self.validation_start = 2000
        
        self.step = 0
        self.env = env
        
        logger.debug(f'Total batch size: {self.total_batch_size}')
        
        # Оптимизировано: Накопитель наград теперь хранится и обрабатывается на CPU через NumPy
        self._episode_rewards = np.zeros((self.env_batch_size, 1))
        
        # Инициализация ключей JAX
        self.rng_key, key0 = jax.device_put(jax.random.split(rng_key, 2), device=self.device)
        self.env_rng_key = jax.device_put(jax.random.split(key0, self.total_batch_size), device=self.device)
        
        self.batched_reset = vmap(self.env.reset)
        self.batched_step = jit(vmap(self.env.step))
        
        self.jit_agent_step = jit(self.agent_step) 
        self.jit_random_step = jit(self.random_step)
        self.jit_reset = jit(self.batched_reset)
        
        self.normalize_obs = True
        
        self.running_obs_mean_std = RunningMeanStd(shape=self.env.observation_space_shape)
        
        if self.normalize_obs:
            norm_path = 'obs-norm/obs_norm_last.npz'
            if os.path.exists(norm_path):
                obs_norm_values = np.load(norm_path)
                self.obs_rms_mean = jax.device_put(jax.numpy.array(obs_norm_values['obs_mean']), device=self.device)
                self.obs_rms_var = jax.device_put(jax.numpy.array(obs_norm_values['obs_var']),  device=self.device)
                if 'obs_count' in obs_norm_values:
                    self.running_obs_mean_std.count = float(obs_norm_values['obs_count'])
            else:
                self.obs_rms_mean = jax.device_put(jax.numpy.zeros(self.env.observation_space_shape), device=self.device)
                self.obs_rms_var = jax.device_put(jax.numpy.ones(self.env.observation_space_shape), device=self.device)
        
        self.episode_reward = deque(maxlen=5)

        self.best_agent_heights = []
        self.best_agent_xs = [] 
        self.best_agent_ys = []

        self.plot_env_idx = self.env_batch_size

        self.local_buffer_size = 16  # Размер "ведра" (накапливаем 16 шагов)
        self.local_obs = []
        self.local_next_obs = []
        self.local_actions = []
        self.local_rewards = []
        self.local_terminated = []
        
    def random_step(self, env_state, inference_key):
        k = 1.00 
        obs_raw = env_state['obs']  # Сохраняем сырое наблюдение
        
        action = jax.random.uniform(
            inference_key, 
            (self.total_batch_size, self.env.action_space_shape[0]),
            minval=k * self.env.ctrlrange_low, 
            maxval=k * self.env.ctrlrange_high
        )
        
        state, true_obs, true_reward, terminated, validation_rewards = self.batched_step(env_state, action)
        
        validation_terminated = jp.zeros(1)[0]
        validation_reward = jp.zeros(1)[0]
        
        # Возвращаем obs_raw и true_obs (сырые) вместо нормализованных
        return state, action, obs_raw, true_obs, true_reward, terminated, validation_terminated, validation_reward

    def agent_step(self, env_state, inference_key, trainee_params, validant_params):
        train_key, val_key = jax.random.split(inference_key, 2)
        
        trainee_func = self.agent_config['trainee_func']
        
        obs_raw = env_state['obs']
        agent_obs = self.normalize_obs_if_needed(obs_raw)  # Нормализуем ТОЛЬКО для инференса актера
        
        action_trainee, _, _ = trainee_func(
            trainee_params, 
            agent_obs[:self.env_batch_size], 
            train_key
        )
        
        action_val, _, _ = trainee_func(
            trainee_params, 
            agent_obs[self.env_batch_size:], 
            val_key
        )
        
        combined_actions = jax.numpy.concatenate([action_trainee, action_val], axis=0)
        
        state, true_obs, true_reward, terminated, validation_rewards = self.batched_step(env_state, combined_actions)
        
        validation_terminated_count = jax.numpy.sum(terminated[self.env_batch_size:])
        validation_reward_sum = jax.numpy.sum(validation_rewards[self.env_batch_size:])
        
        # Возвращаем obs_raw и true_obs (сырые) вместо нормализованных для записи в буфер
        return (state, combined_actions, obs_raw, true_obs, 
                true_reward, terminated, 
                validation_terminated_count, validation_reward_sum)

    def perform_step(self, state0, step_fun, agent_params=None):
        inference_key, self.rng_key = jax.random.split(self.rng_key, 2)

        self.running_obs_mean_std.update(state0['obs'])
        
        if self.normalize_obs:
            self.obs_rms_mean = self.running_obs_mean_std.mean
            self.obs_rms_var = self.running_obs_mean_std.var

        res = step_fun(state0, inference_key, *agent_params) if agent_params is not None else step_fun(state0, inference_key)
        state, action, obs, next_obs, reward, terminated, val_term, val_rew = res
        
        # Выделяем нужные батчи на GPU без копирования на CPU
        obs_gpu = obs[:self.env_batch_size]
        next_obs_gpu = next_obs[:self.env_batch_size]
        action_gpu = action[:self.env_batch_size]
        reward_gpu = reward[:self.env_batch_size]
        terminated_gpu = terminated[:self.env_batch_size]

        # Для подсчета метрик нам все еще нужны награды на CPU.
        # Но это крошечные массивы (всего 1 число на среду), они копируются мгновенно.
        reward_cpu, terminated_cpu = jax.device_get((reward_gpu, terminated_gpu))
        self.compute_metrics_cpu(reward_cpu, terminated_cpu)

        # Накапливаем шаги прямо в видеопамяти (это бесплатно по времени)
        self.local_obs.append(obs_gpu)
        self.local_next_obs.append(next_obs_gpu)
        self.local_actions.append(action_gpu)
        self.local_rewards.append(reward_gpu)
        self.local_terminated.append(terminated_gpu)

        # Если накопили 16 шагов, отправляем их одной транзакцией
        if len(self.local_obs) >= self.local_buffer_size:
            # Соединяем накопленные шаги в одну большую пачку на GPU
            stacked_obs = jax.numpy.concatenate(self.local_obs, axis=0)
            stacked_next_obs = jax.numpy.concatenate(self.local_next_obs, axis=0)
            stacked_actions = jax.numpy.concatenate(self.local_actions, axis=0)
            stacked_rewards = jax.numpy.concatenate(self.local_rewards, axis=0)
            stacked_terminated = jax.numpy.concatenate(self.local_terminated, axis=0)

            # Делаем ОДИН запрос копирования на CPU вместо 16 отдельных!
            obs_cpu, next_obs_cpu, action_cpu, reward_cpu, terminated_cpu = jax.device_get((
                stacked_obs, stacked_next_obs, stacked_actions, stacked_rewards, stacked_terminated
            ))

            # Отправляем готовую большую пачку в буфер
            self.output_queue.put((
                obs_cpu, 
                next_obs_cpu,
                action_cpu,
                reward_cpu, 
                terminated_cpu
            ))

            # Очищаем временные списки
            self.local_obs.clear()
            self.local_next_obs.clear()
            self.local_actions.clear()
            self.local_rewards.clear()
            self.local_terminated.clear()
        
        return state, val_term, val_rew

    def compute_metrics_cpu(self, reward_cpu, terminated_cpu):
        self._episode_rewards += reward_cpu
        
        terminated_count = np.sum(terminated_cpu)
        if terminated_count > 0:
            episode_reward = np.sum(self._episode_rewards * terminated_cpu) / terminated_count
            self.episode_reward.append(float(episode_reward))
            
            # Сброс наград для завершенных эпизодов
            self._episode_rewards *= (1.0 - terminated_cpu)
            
    def write_params_to_file(self, filename, params):
        with open(filename, 'wb') as f:
            f.write(serialization.to_bytes(params))        

    def save_state(self, path, agent_params, qf1_params, qf2_params, q_opt_state, p_opt_state, alpha_opt_state):
        os.makedirs(path, exist_ok=True)
        self.write_params_to_file(os.path.join(path, 'agent.flax'), agent_params)
        self.write_params_to_file(os.path.join(path, 'q1.flax'), qf1_params)
        self.write_params_to_file(os.path.join(path, 'q2.flax'), qf2_params)
        self.write_params_to_file(os.path.join(path, 'q_opt.flax'), q_opt_state)
        self.write_params_to_file(os.path.join(path, 'p_opt.flax'), p_opt_state)
        self.write_params_to_file(os.path.join(path, 'a_opt.flax'), alpha_opt_state)     

    def transfer_params_if_needed(self, params):
            if self.device != self.config['trainer_device']:
                return jax.device_put(params, self.device)
            return params

    def run(self):
        use_norm = '(with obs norm)' if self.normalize_obs else ''
        logger.info(f"Spider-Worker {use_norm} {self.name} alive on {self.device}!")
        self.running = True

        state = self.jit_reset(self.env_rng_key)
        state['step'] = jax.random.randint(self.rng_key, state['step'].shape, 0, self.env.max_steps-2) 
        
        logger.info(f"Starting {self.random_steps_count} random steps to fill buffer...")
        for i in range(self.random_steps_count):
            state, _, _ = self.perform_step(state, self.jit_random_step)
            if i % 100 == 0 and len(self.episode_reward) > 0:
                # Оптимизировано: Среднее значение через быстрый numpy на CPU
                episode_reward = np.mean(self.episode_reward)
                logger.info(f'Warm-up {i:05d}\t средняя награда: {episode_reward:1.4f}')
                
        logger.info('Random steps finished. Starting training loop.')
        
        self.step = 0
        
        trainee_name, params, qf1, qf2, global_step, opt_params = self.agent_queue.get()
        trainee_params = self.transfer_params_if_needed(params)
        
        validant_params = trainee_params
        validant_qf1, validant_qf2 = qf1, qf2
        validant_opt_params = opt_params 
        
        total_validation_reward = 0.0
        total_episodes = 0
        validation_step = 0
        
        while self.running: 
            if self.step % 16 == 0:
                trainee_name, trainee_params, qf1, qf2, global_step, opt_params = self.agent_queue.get()
                trainee_params = self.transfer_params_if_needed(trainee_params)
            
            state, val_term, val_rew = self.perform_step(
                state, 
                self.jit_agent_step, 
                (trainee_params, validant_params)
            )
            
            # Оптимизировано: Переносим координаты и шаг ОДНИМ запросом вместо 4 блокировок
            com_val, step_val = jax.device_get((
                state['last_com'][self.plot_env_idx], 
                state['step'][self.plot_env_idx]
            ))
            
            best_agent_x = float(com_val[0])
            best_agent_y = float(com_val[1])
            best_agent_z = float(com_val[2])
            is_reset = (int(step_val) == 0)
            
            self.best_agent_xs.append(best_agent_x)
            self.best_agent_ys.append(best_agent_y)
            self.best_agent_heights.append(best_agent_z)
            
            if is_reset and len(self.best_agent_heights) > 100:
                role_name = "Ученик (Exploration)" if self.plot_env_idx < self.env_batch_size else "Тестировщик (Validation)"
                logger.warning(f"Отрисован график для робота №{self.plot_env_idx} ({role_name})")
                
                self.plot_and_save_trajectory_graphs()
                
                self.best_agent_heights = []
                self.best_agent_xs = []
                self.best_agent_ys = []
                
                self.plot_env_idx = np.random.randint(0, self.total_batch_size)

            if self.step % 5000 == 0:
                mean = self.running_obs_mean_std.mean
                var = self.running_obs_mean_std.var
                count = self.running_obs_mean_std.count
                
                os.makedirs('obs-norm', exist_ok=True)
                np.savez('obs-norm/obs_norm_last.npz', 
                        obs_mean=jax.device_get(mean), 
                        obs_var=jax.device_get(var),
                        obs_count=float(count))
            
            self.step += 1 
            
            if self.step >= self.validation_start:
                total_validation_reward += val_rew
                total_episodes += val_term
                validation_step += 1  
            
            if validation_step >= self.validation_steps and total_episodes > 0:
                reward_per_episode = float(total_validation_reward / total_episodes)
                self.validant_last_reward = reward_per_episode
                
                if self.validant_best_reward is None or reward_per_episode > self.validant_best_reward:
                    self.validant_best_reward = reward_per_episode

                    agent_path = os.path.join('checkpoints', f'{trainee_name}', f'best_at_{global_step}')             
                    logger.info(f"!!! NEW RECORD: {reward_per_episode:1.2f}m. Saving best agent to '{agent_path}'")
                    self.save_state(agent_path, trainee_params, qf1, qf2, *opt_params)

                    np.savez(os.path.join(agent_path, 'obs_norm.npz'), 
                            obs_mean=jax.device_get(self.running_obs_mean_std.mean), 
                            obs_var=jax.device_get(self.running_obs_mean_std.var),
                            obs_count=float(self.running_obs_mean_std.count))
                    
                    validant_params = trainee_params
                    validant_opt_params = opt_params
                    validant_qf1, validant_qf2 = qf1, qf2
                
                total_validation_reward = 0.0
                total_episodes = 0
                validation_step = 0
                
        logger.debug(f"Worker finished")

    def normalize_obs_if_needed(self, obs):
        if not self.normalize_obs:
            return obs
        return jax.numpy.clip((obs - self.obs_rms_mean) / jax.numpy.sqrt(self.obs_rms_var + 1e-8), -10.0, 10.0)

    def plot_and_save_trajectory_graphs(self):
        try:
            import matplotlib
            matplotlib.use('Agg')  
            import matplotlib.pyplot as plt
            
            fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            fig.suptitle(f'Траектория лучшего паука (Шаг обучения: {self.step})', fontsize=14, fontweight='bold')
            
            axs[0].plot(self.best_agent_heights, label='Высота корпуса (Z)', color='red', linewidth=2)
            axs[0].axhline(y=0.15, color='green', linestyle='--', label='Идеальная высота (~0.15м)')
            axs[0].axhline(y=0.08, color='black', linestyle=':', label='Порог смерти (0.08м)')
            axs[0].set_ylabel('Высота Z (метры)')
            axs[0].set_ylim(0.0, 0.4)
            axs[0].grid(True, linestyle=':', alpha=0.6)
            axs[0].legend(loc='upper right')
            
            axs[1].plot(self.best_agent_xs, label='Пройденное расстояние (X)', color='blue', linewidth=2)
            axs[1].set_ylabel('Расстояние X (метры)')
            axs[1].grid(True, linestyle=':', alpha=0.6)
            axs[1].legend(loc='upper left')
            
            axs[2].plot(self.best_agent_ys, label='Отклонение вбок (Y)', color='purple', linewidth=2)
            axs[2].axhline(y=0.0, color='black', linestyle='--', alpha=0.5, label='Центр коридора (Y=0)')
            axs[2].set_ylabel('Смещение Y (метры)')
            axs[2].set_xlabel('Шаги внутри эпизода')
            axs[2].set_ylim(-1.0, 1.0)  
            axs[2].grid(True, linestyle=':', alpha=0.6)
            axs[2].legend(loc='upper left')
            
            plt.tight_layout()
            
            os.makedirs('plots', exist_ok=True)
            plt.savefig('plots/height_trajectory_latest.png', dpi=150)
            plt.close()
            
        except Exception as e:
            logger.error(f"Ошибка при рисовании графиков траектории: {e}")