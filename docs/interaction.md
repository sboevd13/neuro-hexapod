# Learning from object interaction in MuJoCo

Experimental branch `feature/object-affordances-v1`, built on the object-curiosity experiment.

## Goal

Instead of only mapping and looking at obstacles, the simulated hexapod executes
**one bounded forward probe** against a physical MuJoCo box and measures whether
its observed surface moved. The same learned category preference is used when
faced with a new box of similar apparent width.

- `train_light_narrow`: freejoint, low-mass obstacle; the robot tries a push.
- `train_fixed_wide`: anchored obstacle; the robot tries once and learns.
- `heldout_light_narrow`: new light object, same observed category.
- `heldout_fixed_wide`: new anchored object, same observed category.

The policy sees only idealized 23-beam lidar returns and whether an experimental
contact switch was triggered. It **does not** read freejoint metadata, mass,
known labels, or true object displacement. Those are used only by the
simulator/sensor adapter or the independent evaluator. The visual width classes
(`narrow`, `wide`) are coarse range-sensor proxies and not real object recognition.

## How to run

```bash
python interact_laptop.py --seed 7 --seconds 8 --output runs/interaction
# For visual inspection (four sequential MuJoCo scenes):
python interact_laptop.py --seed 7 --seconds 8 --view --output runs/interaction/view
```

Outputs: `runs/interaction/affordance_memory.json` (versioned experience),
`runs/interaction/report.json` (sensor motion **and** privileged truth values
clearly separated). The model stays isolated from existing trained PPO, V5,
frontier exploration, and curiosity settings.

The learning algorithm uses Beta-smoothed estimates of pushability by observed
width category. Initial unknown objects permit a limited test, then the
posterior mobility estimate biases future `observe` versus `probe` decisions.
This is **interpretable online learning**, not a deep neural network or a
universal affordance predictor. The four-case trial is a controlled demonstration
only and does not show manipulation in arbitrary scenes.

## Limitations

- The probe uses ordinary V5 forward locomotion: there is no arm, grasping,
  precision force control, or learned 18-joint pushing skill.
- The robot's world position is obtained directly from MuJoCo, not SLAM.
- Ray intersections use simplified oriented-square geometry; rotating boxes
  may be imperfectly estimated.
- MuJoCo's contact state acts as an **idealized tactile switch**; real robots
  would need physical force/touch feedback and a watchdog.
- Only simulator trials, no physical robot actuation.
- Safety of pushing household objects cannot be inferred from sandbox trials.

## Проверенные результаты (GitHub Actions, 19 ноября 2026)

[Запуск CI на шести seed](https://github.com/sboevd13/neuro-hexapod/actions/runs/44542470781)
подтвердил **24 физические пробы** на четырёх типах предъявлений.
Шесть seed изменяют начальное состояние сервоприводов; это не шесть разных
планировок комнат. Робот работает в MuJoCo с базовой походкой V5.

| Объект | Действие | Результат в среднем |
| --- | --- | --- |
| Первый лёгкий узкий ящик | Пробный толчок | Передвинулся на **7,9 см** |
| Неподвижный широкий ящик | Пробный толчок | Сдвиг **0 см** |
| Новый лёгкий узкий ящик | Толчок на основе опыта | Передвинулся на **14,8 см** |
| Новый неподвижный широкий ящик | Наблюдение без толчка | Контакта не было |

Во всех прогонах: **0 падений**, 12 успешных сдвигов мобильных ящиков,
6 решений не трогать повторно встретившийся тип неподвижного препятствия.
Измеренное дальномером смещение сравнивалось с независимой физической истиной
из MuJoCo, которую контроллер при выборе действия не видит.

Стратегия не распознаёт категорию «коробка» и не знает свойства материалов.
Она накапливает статистику реакции на пробный контакт для *двух грубых
размерных групп*. Начальные тестовые условия намеренно соотнесли маленькие
и подвижные, большие и неподвижные объекты. Для произвольных размеров и
непредвиденных предметов полученные значения достоверность не гарантируют.

Протокол с показателями по каждому seed: `validation/object_affordances_v1.json`.
