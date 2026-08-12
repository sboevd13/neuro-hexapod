import logging
import os
import jax
from queue import Queue
from flax import serialization
import numpy as np

# Импортируем все наши модули
from worker import WorkerThread
from buffer import BufferThread
from trainer import TrainingThread
from arena import BattleArena
from agent import ActorSimple_skip

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    level=logging.WARNING,
    force=True
)

# Определение устройств (GPU/CPU)
devices = jax.devices()
print(f"Доступные устройства: {devices}")
worker_device = devices[0]
trainer_device = devices[1] if len(devices) > 1 else worker_device

# --- ГЛОБАЛЬНЫЙ КОНФИГ ОБУЧЕНИЯ ---
config = dict(
    seed = 42,
    worker_device=worker_device,
    trainer_device=trainer_device,

    # Сколько пауков бегает одновременно
    worker_batch_size=256,  # Обучаются
    validation_batch_size=32, # Сдают экзамен (всего будет 512+64 = 576 сред)
    
    trainer_batch_size=1024, # Размер пачки данных для обучения нейросети

    buffer_size=1_000_000,  # Размер памяти (шагов)
    random_steps_count=1000, # Сколько шагов делать случайно в начале для прогрева
    
    # Гиперпараметры SAC
    initial_alpha = 1.0,
    autotune_alpha = True,
    tau = 0.005,
    gamma = 0.985,
    q_lr = 0.0003, # Немного снизил для стабильности 18-ти моторов
    p_lr = 0.0003,
    
    total_steps=2_000_000, # Общее количество обновлений нейросети
    warmup_steps=10_000,   # Шаги, за которые скорость обучения вырастет до рабочей
    
    report_to_tensorboard=True, # Включаем логи (можно смотреть через tensorboard --logdir runs)
    report_to_wandb=False       # Выключил по умолчанию, если нет аккаунта
)

# Инициализация ключей случайности
_, worker_key, trainer_key, init_key = jax.random.split(jax.random.key(config['seed']), 4)

# 1. Создаем симуляцию (тело)
env = BattleArena()

# 2. Создаем нейросеть (мозг)
agent_skip = ActorSimple_skip(
    env.action_space_shape[0], # 18 моторов
    env.ctrlrange_high, 
    env.ctrlrange_low,
    512, 512
) 

# Инициализируем параметры весов
state0 = env.reset(jax.random.key(0))
obs = state0['obs'] # ИСПРАВЛЕНО: убрали [0]
observations_count = jax.numpy.prod(jax.numpy.array(obs.shape)).item()
agent_params = agent_skip.init(init_key, jax.numpy.ones((1, observations_count)))['params']

# Расчет общего количества параллельных сред
total_batch_size = config['worker_batch_size'] + config['validation_batch_size']

# 3. Запускаем Буфер памяти
bt = BufferThread(
    config['buffer_size'], 
    config['worker_batch_size'], 
    env.observation_space_shape, 
    env.action_space_shape
)

# Очередь для передачи весов от тренера воркеру
worker_agent_queue = Queue(maxsize=8)

# 4. Запускаем Воркера (сборщика опыта)
wt = WorkerThread(
    config=config,
    env=env, 
    rng_key=worker_key, 
    result_queue=bt.input_queue, 
    agent_queue=worker_agent_queue, 
    agent_config=dict(
        trainee_func=agent_skip.get_action,
        validation_agent_func=agent_skip.get_action,
        ref_agent_params=[], # Пусто, врагов нет
        validation_agent_params=[] # Пусто, будем использовать trainee_params
    ),
    name='Spider-Worker-0'
)

# 5. Запускаем Тренера (обновителя мозгов)
tt = TrainingThread(
    config=config,
    buffer_thread=bt,
    agent_queue=worker_agent_queue,
    agent=agent_skip,
    agent_params=agent_params,
    rng_key=trainer_key,
    observation_space_shape=env.observation_space_shape,
    action_space_shape=env.action_space_shape,
    worker_thread=wt,
    name='Spider-Trainer'
)

# --- ПУСК ---
# Проверяем, есть ли сохраненный чекпоинт в папкеUnnamed, чтобы продолжить обучение
resume_path = 'good_modes/last'

if os.path.exists(resume_path):
    print("\n" + "="*50)
    print("-> ОБНАРУЖЕН ПРЕДЫДУЩИЙ ЧЕКПОИНТ!")
    print(f"-> Загружаю веса, критиков и оптимизаторы из: {resume_path}")
    
    # Загружаем полное состояние обучения
    tt.load_state(resume_path)
    
    norm_file = os.path.join(resume_path, 'obs_norm_last.npz')
    if os.path.exists(norm_file):
        print(f"-> Загружаю статистику нормализации датчиков: {norm_file}")
        obs_norm_values = np.load(norm_file)
        
        # Переносим массивы на JAX-девайс воркера
        mean_jax = jax.device_put(jax.numpy.array(obs_norm_values['obs_mean']), wt.device)
        var_jax = jax.device_put(jax.numpy.array(obs_norm_values['obs_var']), wt.device)
        
        # Записываем загруженные данные напрямую в "мозг" воркера
        wt.running_obs_mean_std.mean = mean_jax
        wt.running_obs_mean_std.var = var_jax
        wt.obs_rms_mean = mean_jax
        wt.obs_rms_var = var_jax
        
        # Восстанавливаем счетчик шагов нормализации
        if 'obs_count' in obs_norm_values:
            wt.running_obs_mean_std.count = float(obs_norm_values['obs_count'])
            print(f"-> Счетчик шагов нормализации успешно восстановлен: {wt.running_obs_mean_std.count}")
        else:
            wt.running_obs_mean_std.count = 1_000_000.0
            print("-> Предупреждение: obs_count не найден. Установлен фоллбек 1,000,000.0.")
            
        print("-> Нормализация датчиков успешно восстановлена!")
    else:
        print("-> Предупреждение: obs_norm_last.npz не найден в папке чекпоинта. Будет использована дефолтная статистика.")
    
    # Синхронизируем шаги воркера с тренером
    if wt is not None:
        wt.step = tt.current_step
        
    print(f"-> Состояние успешно восстановлено. Продолжаем обучение с шага {tt.current_step}!")
    print("="*50 + "\n")
else:
    print("\n" + "="*50)
    print("-> Предыдущий чекпоинт не найден. Начинаем обучение с нуля.")
    print("="*50 + "\n")
# ==========================================
# === ЗАПУСК ПОТОКОВ ===
# ==========================================
print("Запуск системы обучения паука-спасателя...")
bt.start()
wt.start()
tt.start()

try:
    tt.join() # Ждем окончания (2 млн шагов)
except KeyboardInterrupt:
    print("\nОстановка обучения пользователем...")
    wt.running = False
    tt.running = False


# ==========================================
# === СОХРАНЕНИЕ ФИНАЛЬНОГО РЕЗУЛЬТАТА ===
# ==========================================
path_to_save = 'checkpoints/final_quadruped_model'

# Сохраняем финальные веса
tt.save_state(path_to_save)

# Сохраняем финальную нормализацию датчиков
os.makedirs(path_to_save, exist_ok=True)
np.savez(
    os.path.join(path_to_save, 'obs_norm.npz'), 
    obs_mean=jax.device_get(wt.running_obs_mean_std.mean), 
    obs_var=jax.device_get(wt.running_obs_mean_std.var),
    obs_count=float(wt.running_obs_mean_std.count)
)

print(f"\nОбучение завершено. Итоговая модель и obs_norm.npz сохранены в {path_to_save}")