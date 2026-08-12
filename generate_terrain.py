import numpy as np
from PIL import Image, ImageFilter
import os

def generate_rubble_terrain():
    # Размер сетки 300x100 (10 пикселей на 1 метр, итого 30x10 метров)
    width, height = 300, 100
    terrain = np.zeros((height, width), dtype=np.float32)
    
    # Задаем максимальные физические высоты (при условии, что в XML max hfield height = 0.1м (10 см))
    # Значения нормированы: высота в метрах / 0.1
    h_gravel = 0.008 / 0.1  # Мелкий шум (галька): 0.8 см
    h_mound  = 0.025 / 0.1  # Невысокие плавные кочки: до 2.5 см
    h_max    = 0.030 / 0.1  # Абсолютный предел высоты: 3.0 см

    # ----------------------------------------------------
    # ШАГ 1: Равномерный мелкий шум (галька) по всей карте
    # ----------------------------------------------------
    gravel_noise = np.random.uniform(0.0, h_gravel, (height, width))
    
    # Превращаем в мягкую базовую текстуру земли
    base_img = Image.fromarray((gravel_noise * 255).astype(np.uint8))
    base_img_smoothed = base_img.filter(ImageFilter.GaussianBlur(radius=0.5))
    terrain += np.array(base_img_smoothed, dtype=np.float32) / 255.0

    # ----------------------------------------------------
    # ШАГ 2: Равномерная генерация пологих кочек по всей карте
    # ----------------------------------------------------
    # Разбрасываем 180 небольших сглаженных кочек по всей площади арены
    num_mounds = 180
    mounds = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(num_mounds):
        # Координаты центра кочки
        rx = np.random.randint(4, width - 4)
        ry = np.random.randint(4, height - 4)
        
        # Радиус кочки в пикселях (от 30 до 80 см)
        size = np.random.randint(3, 8)
        
        # Высота конкретной кочки
        h = np.random.uniform(0.005 / 0.1, h_mound)
        
        # Генерация маски с косинусным сглаживанием краев
        y, x = np.ogrid[-ry:height-ry, -rx:width-rx]
        dist = np.sqrt(x*x + y*y)
        mask = dist <= size
        
        falloff = (1.0 + np.cos(np.pi * dist[mask] / size)) * 0.5
        mounds[mask] = np.maximum(mounds[mask], h * falloff)
        
    # Объединяем мелкий шум и плавные кочки
    terrain = np.maximum(terrain, mounds)

    # ----------------------------------------------------
    # ШАГ 3: Ограничение высоты и финальное сглаживание
    # ----------------------------------------------------
    terrain = np.clip(terrain, 0.0, h_max)
    
    final_raw_img = Image.fromarray((terrain * 255).astype(np.uint8))
    # GaussianBlur делает переходы высот плавными, чтобы коллизии робота с землей были стабильными
    final_smoothed_img = final_raw_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Сохраняем готовую карту высот
    os.makedirs('models', exist_ok=True)
    final_smoothed_img.save("models/terrain.png")
    print("Однородная низкая карта высот 300x100 успешно сохранена в models/terrain.png.")

if __name__ == "__main__":
    generate_rubble_terrain()