# create_kaggle_zip.py
import zipfile
from pathlib import Path

def create_kaggle_zip(folder_name, output_name):
    """Crée un ZIP avec chemins compatibles Kaggle (forward slashes)"""
    
    folder = Path(folder_name)
    output = Path(output_name)
    
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in folder.rglob('*'):
            if file_path.is_file():
                # ✅ as_posix() convertit \ en / pour Linux
                arcname = file_path.relative_to(folder).as_posix()
                zipf.write(file_path, arcname)
                print(f"✅ {arcname}")
    
    print(f"\n🎉 ZIP créé : {output.resolve()}")

# Usage
create_kaggle_zip('data_propre', 'data_propre_kaggle.zip')