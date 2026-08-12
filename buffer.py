import threading
from queue import Queue
import time
import numpy as np
import jax  # Добавлено для оптимизации переноса данных на GPU

class BufferThread(threading.Thread):
    def __init__(self, buffer_size, env_batch_size, observation_space_shape, action_space_shape, name='Buffer Thread'):
        super(BufferThread, self).__init__(name=name)
        self.daemon = True
        self.input_queue = Queue()
        self.running = False
        
        # Блокировка для безопасного доступа к буферу из разных потоков
        self.lock = threading.Lock()
        
        self.buffer = ReplayBuffer(buffer_size, env_batch_size, observation_space_shape,
                                   action_space_shape)
        self.output_count = 0
        self.total_replay_samples = 0
        
        # Инициализация переменных SPS во избежание AttributeError при раннем чтении
        self._input_sps = 0.0
        self._output_sps = 0.0

    @property
    def input_sps(self):
        return self._input_sps

    @property
    def output_sps(self):
        return self._output_sps

    def stop(self):
        self.running = False

    def run(self) -> None:
        self.running = True
        iteration = 0
        input_count = 0
        last_t = time.time()

        while self.running:
            obs, next_obs, action, reward, terminated = self.input_queue.get(block=True)
            
            # Запись в буфер защищена блокировкой
            with self.lock:
                self.buffer.add(obs, next_obs, action, reward, terminated)
                
            input_count += obs.shape[0]
            self.total_replay_samples += obs.shape[0]
            dt = time.time() - last_t
            if dt >= 1.0:
                self._input_sps = input_count / dt
                self._output_sps = self.output_count / dt
                input_count = 0
                self.output_count = 0
                last_t = time.time()

            iteration += 1

    def sample(self, batch_size):
        self.output_count += batch_size
        
        # Чтение из буфера защищено блокировкой
        with self.lock:
            numpy_batch = self.buffer.sample(batch_size)
            
        # Асинхронный перенос батча на GPU/TPU устройство
        return jax.device_put(numpy_batch)


class ReplayBuffer:
    def __init__(self, size, env_batch_size, observation_space_shape, action_space_shape):
        self.size = int(np.ceil(size / env_batch_size)) * env_batch_size
        self.env_batch_size = env_batch_size
        self.obs_buffer = np.zeros((self.size, *observation_space_shape))
        self.next_obs_buffer = np.zeros_like(self.obs_buffer)
        self.action_buffer = np.zeros((self.size, *action_space_shape))
        self.reward_buffer = np.zeros((self.size, 1))
        self.terminated_buffer = np.zeros((self.size, 1))
        self.ptr = 0
        self.full = False
        
        # Генератор случайных чисел инициализируется один раз
        self.rng = np.random.default_rng()

    def add(self, obs, next_obs, action, reward, terminated):
        n = obs.shape[0]  # Размер входящей пачки (теперь может быть равен 16 * env_batch_size)
        
        # Циклический буфер: проверяем, помещается ли пачка до конца массива
        if self.ptr + n <= self.size:
            self.obs_buffer[self.ptr:self.ptr + n] = obs
            self.next_obs_buffer[self.ptr:self.ptr + n] = next_obs
            self.action_buffer[self.ptr:self.ptr + n] = action
            self.reward_buffer[self.ptr:self.ptr + n] = reward
            self.terminated_buffer[self.ptr:self.ptr + n] = terminated
            self.ptr += n
        else:
            # Если пачка не влезает, делим её на две части: 
            # первую пишем до конца массива, вторую — в начало массива
            space_left = self.size - self.ptr
            
            self.obs_buffer[self.ptr:] = obs[:space_left]
            self.obs_buffer[:n - space_left] = obs[space_left:]
            
            self.next_obs_buffer[self.ptr:] = next_obs[:space_left]
            self.next_obs_buffer[:n - space_left] = next_obs[space_left:]
            
            self.action_buffer[self.ptr:] = action[:space_left]
            self.action_buffer[:n - space_left] = action[space_left:]
            
            self.reward_buffer[self.ptr:] = reward[:space_left]
            self.reward_buffer[:n - space_left] = reward[space_left:]
            
            self.terminated_buffer[self.ptr:] = terminated[:space_left]
            self.terminated_buffer[:n - space_left] = terminated[space_left:]
            
            self.ptr = n - space_left
            self.full = True

    def sample(self, batch_size):
        p = self.ptr if not self.full else self.size
        
        # Оптимизированный выбор случайных индексов через integers вместо choice
        idx = self.rng.integers(0, p, size=batch_size)
        
        return (
            self.obs_buffer[idx], 
            self.next_obs_buffer[idx], 
            self.action_buffer[idx], 
            self.reward_buffer[idx], 
            self.terminated_buffer[idx]
        )