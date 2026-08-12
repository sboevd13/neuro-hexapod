import os
from pathlib import Path

def combine_selected_files(project_dir, output_file_path):
    # Явно исключаем распространенные папки окружений, сборщиков и библиотек
    exclude_dirs_exact = {
        'bin', 'obj', 'node_modules', '__pycache__', 
        'venv', 'env', 'lib', 'lib64', 'include', 'share',
        'build', 'dist', 'eggs', 'parts', 'develop-eggs'
    }

    project_path = Path(project_dir).resolve()
    output_path = Path(output_file_path).resolve()
    current_script_path = Path(__file__).resolve()

    opened_files_count = 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(project_path):
            # 1. Исключаем все скрытые папки (начинающиеся с '.')
            # 2. Исключаем папки из черного списка по точному совпадению
            dirs[:] = [
                d for d in dirs 
                if not d.startswith('.') and d not in exclude_dirs_exact
            ]

            for file in files:
                file_path = Path(root) / file
                
                if file_path == output_path or file_path == current_script_path:
                    continue

                is_target_file = (file_path.suffix == '.py') or (file_path.name == 'arena.xml')

                if is_target_file:
                    relative_path = file_path.relative_to(project_path)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                        
                        outfile.write(f"=========================================\n")
                        outfile.write(f"FILE: {relative_path}\n")
                        outfile.write(f"=========================================\n\n")
                        outfile.write(content)
                        outfile.write("\n\n")
                        
                        # Выводим в консоль добавленный файл для контроля
                        print(f"Добавлен: {relative_path}")
                        opened_files_count += 1
                        
                    except (UnicodeDecodeError, PermissionError) as e:
                        print(f"Пропущен (ошибка чтения): {relative_path}")

    print(f"\nГотово! Обработано файлов: {opened_files_count}")
    print(f"Результат сохранен в: {output_path}")

if __name__ == "__main__":
    PROJECT_DIRECTORY = "." 
    OUTPUT_FILE = "combined_project_code.txt"
    
    combine_selected_files(PROJECT_DIRECTORY, OUTPUT_FILE)