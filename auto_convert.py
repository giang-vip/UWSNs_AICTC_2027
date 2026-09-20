import os
import json
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from uwsn import config

INPUT_DIR = "data"
OUTPUT_DIR = "Kmean_Data"

def process_file(txt_path):
    rel_path = os.path.relpath(txt_path, INPUT_DIR)
    dir_name = os.path.dirname(rel_path)
    base_name = os.path.basename(txt_path).replace('.txt', '')
    
    # Bỏ qua các file không nằm trong thư mục Distribution
    if "Distribution" not in dir_name:
        return
        
    # Tạo 3 thư mục riêng biệt: txt, json và images
    txt_dir = os.path.join(OUTPUT_DIR, dir_name, "txt")
    json_dir = os.path.join(OUTPUT_DIR, dir_name, "json")
    img_dir = os.path.join(OUTPUT_DIR, dir_name, "images")
    
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    new_txt_path = os.path.join(txt_dir, f"{base_name}.txt")
    json_path = os.path.join(json_dir, f"{base_name}.json")
    png_path = os.path.join(img_dir, f"anh_{base_name}.png")
    
    if os.path.exists(json_path) and os.path.exists(png_path) and os.path.exists(new_txt_path):
        return

    # 1. Copy file .txt gốc sang thư mục txt
    shutil.copy(txt_path, new_txt_path)

    # 2. Đọc dữ liệu và tạo file .json
    nodes = []
    xs, ys, zs = [], [], []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for idx, line in enumerate(lines[1:]):
        parts = line.strip().split()
        if len(parts) >= 3:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            nodes.append({
                "id": idx + 1,
                "x": x, "y": y, "z": z,
                "initial_energy": config.INITIAL_ENERGY,
                "residual_energy": config.INITIAL_ENERGY
            })
            xs.append(x)
            ys.append(y)
            zs.append(z)
            
    with open(json_path, 'w') as f:
        json.dump(nodes, f, indent=4)
        
    # 3. Vẽ và lưu ảnh đồ thị
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='blue', marker='o', s=15, alpha=0.5, label='Sensors')
    ax.scatter([1000], [1000], [1500], c='red', marker='^', s=100, label='Base Station')
    
    ax.set_title(f'Phân bố không gian - {base_name}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Đã xử lý xong: {base_name}")

if __name__ == "__main__":
    print("Bắt đầu xử lý dữ liệu...")
    txt_files = glob.glob(os.path.join(INPUT_DIR, '**', '*.txt'), recursive=True)
    
    for file in txt_files:
        process_file(file)
        
    print("\nHoàn thành! Toàn bộ file txt, json và ảnh đã được phân loại gọn gàng.")