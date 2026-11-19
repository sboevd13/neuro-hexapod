# Камера, виртуальные датчики и оценка положения в MuJoCo

Это **экспериментальный симуляционный режим**. Он не заменяет эталонную походку V5 и не подключается к физическому роботу. Пользователь может самостоятельно проверить автономную разведку с ошибками сенсоров и сравнить её с режимом идеальных координат.

## Что добавлено

- **Реальная RGB-камера MuJoCo**, закреплённая на корпусе (разрешение 128×96), и карта глубины для измерения расстояния до видимых поверхностей. Кадры сохраняются как `camera_*.png`.
- **Визуальная сегментация** по единой бирюзовой окраске тестовых предметов. Камера видит реальные пиксели рендеринга, а не получает ID объекта, название, массу или координаты предмета. Этот детектор **не распознаёт любые произвольные объекты**: он работает только при контролируемой цветовой разметке.
- **IMU (гироскоп)** с шумом и **одометрия по опорным ногам**: позиции лап относительно корпуса вычисляются из кинематики, контакты с полом — из MuJoCo. При недостатке надёжных опор используется приближённая модель скорости походки.
- **Дальномер** (23 горизонтальных луча с шумом и пропусками), который строит карту. Для его эмуляции датчиков используется физическая геометрия симулятора, но система навигации получает только измеренные расстояния, а не точные координаты объектов или робота.
- **Локальное сопоставление сканов с картой** для частичной коррекции накопленной ошибки одометрии. Это двухмерная карта + коррекция местоположения, но **не полноценный визуальный SLAM** с отслеживанием признаков камеры, релоакализацией и замыканием петель.
- **Подключение к автономному планировщику и любопытству**: для вычисления маршрута используются оценённые (а не настоящие) координаты робота. Сегментация RGB подтверждает наличие объектов перед инициированием осмотра.

**Важно:** точные координаты MuJoCo используются в драйвере для **эмуляции измерений датчиков** и для расчёта **ошибки в отчёте**, но не передаются в алгоритм навигации в штатном режиме `estimated`. Исключение — специально обозначенный контрольный режим `oracle` для сравнения.

## Запуск

```bash
cd ~/arena
source .venv-laptop/bin/activate
python visual_autonomy.py --seed 7 --seconds 60 --view --output runs/visual_demo
```

Можно запустить в headless-режиме без `--view`, если настроен offscreen OpenGL. При использовании Linux и серверного процессора обычно требуется `MUJOCO_GL=egl` и доступный EGL/OpenGL. На GUI-хосте можно оставить системный backend.

Контрольные режимы:

```bash
python visual_autonomy.py --localization estimated --seed 7 --seconds 60 --output runs/with_matching
python visual_autonomy.py --localization odometry --seed 7 --seconds 60 --output runs/without_matching
python visual_autonomy.py --localization oracle --seed 7 --seconds 60 --output runs/with_truth
```

Выходы: `metrics.json` (ошибка позы, покрытие карты, количество кадров камеры, визуальных обнаружений и исправлений карты), `map.json`, `objects.json`, `visual_detections.json` и RGB-кадры `camera_*.png`.

## Методика валидации

1. Прогон нескольких независимых комнат с одинаковыми random seed и V5 эталонной походкой.
2. Проверка наличия реальных RGB-кадров MuJoCo, обнаружений окрашенных предметов, данных IMU и одометрии опорных ног, роста карты.
3. Сравнение `position_rmse_m` с оценкой по одной одометрии и идеальной позой только **в отчёте**. Нельзя утверждать, что сопоставление сканов улучшает точность, пока это не показано экспериментально.
4. Запись падений и физических контактов; отсутствие происшествий в конкретных сценах не доказывает безопасность.

## Что пока отсутствует

Нет полноценной визуальной одометрии по кадрам (optical flow/ORB), настоящего визуально-инерциального SLAM, калиброванной глубины и модели шума реального оборудования, достоверной детекции произвольных предметов, переноса на реальную камеру и интеграции физического толкания в сенсорный режим. Корректнее называть этот этап **RGB-assisted LiDAR localization + proprioceptive odometry**. Следующие этапы: стабильная визуальная одометрия, замыкание петель, перенести предыдущий модуль взаимодействия на оценённую позу.

## Интеграция с физическими экспериментами в одной симуляции

В отличие от отдельной проверки `visual_autonomy.py`, существующий
`autonomous_world.py` теперь поддерживает флаг `--vision-odometry`: камера
видит объекты, локализация оценивается по IMU/ногам/сканам, а обычная
логика любопытства и ограниченного толкания может обновлять нейросеть.

```bash
python autonomous_world.py --seed 11 --seconds 90 --vision-odometry \
  --experiment-model models/active_displacement_ensemble.json \
  --experiment-experience models/active_push_replay.json \
  --view --output runs/full_visual_experiment
```

Кадры камеры сохраняются в `rgb_first.png` и `rgb_last.png`, а
`report.json` фиксирует ошибку положения, число контактов, предсказания
и число фактических обновлений модели. Режим использует 3 физических объекта
со случайной массой и трением, но с одинаковой яркой окраской. Включение
флага **не превращает** лабораторную сегментацию цветов в универсальное зрение.
Он не предназначен для работы с настоящим роботом до появления датчиков силы,
аппаратного стопа и проверки ограничений.


## Measured MuJoCo results (19 November 2026)

[Reproducible CI evidence](https://github.com/sboevd13/neuro-hexapod/actions/runs/44563168317),
including actual RGB frames as artifacts.

| Room seed | Pose RMSE (scan matching) | Pose RMSE (foot + gyro only) | RGB detections | Map coverage |
| --- | ---: | ---: | ---: | ---: |
| 7 | 10.42 cm | 6.12 cm | 42 | 56.75% |
| 11 | 4.18 cm | 8.65 cm | 24 | 53.88% |
| 23 | 5.49 cm | 5.65 cm | 47 | 55.63% |
| **Mean** | **6.70 cm** | **6.81 cm** | — | — |

Each room ran for 22 seconds (V5 gait). In total 75 camera frames, eight
object inspections, zero robot/object contacts and zero falls.

Three additional **continuous physics and learning** sessions: seed 11
(55 s), seed 7 (80 s), seed 23 (80 s), with 43 camera frames each:
four actual object contacts, three additional gentle push actions,
three sensor-outcome labels and three in-session updates of the neural
outcome ensemble, and no falls. Pose RMSE across these longer sessions
was 14.20, 11.77 and 13.87 cm respectively.

The scan correction is conservative but **not a proven improvement**:
on seed 7 it was less accurate than odometry alone. The short independent
scenes are a smoke test, not a statistically strong reliability result,
and the absence of falls is not a safety guarantee. Full structured
evidence: `validation/visual_localization_v1.json`.
