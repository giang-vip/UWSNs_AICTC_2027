import json
import random
import os

# Cấu hình
SENSOR_COUNTS = [35, 40, 45, 50, 55, 60]
SPACE_SIZE = 400
INITIAL_ENERGY = 100.0

# Thư mục lưu dữ liệu (Tương thích với cấu trúc của runner.py)
output_dir = os.path.join("data", "nodes-data")
os.makedirs(output_dir, exist_ok=True)

def generate_nodes(count):
    nodes = []
    for i in range(1, count + 1):
        # Rải tọa độ ngẫu nhiên từ 10 đến 390 để tránh nằm sát rìa
        node = {
            "id": i,
            "x": random.randint(10, SPACE_SIZE - 10),
            "y": random.randint(10, SPACE_SIZE - 10),
            "z": random.randint(10, SPACE_SIZE - 10),
            "initial_energy": INITIAL_ENERGY,
            "residual_energy": INITIAL_ENERGY
        }
        nodes.append(node)
    return nodes

def main():
    print(f"{'='*40}")
    print("BẮT ĐẦU SINH DỮ LIỆU MÔ PHỎNG UWSN")
    print(f"{'='*40}")
    
    for count in SENSOR_COUNTS:
        nodes = generate_nodes(count)
        filename = f"nodes_{count}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=4, ensure_ascii=False)
            
        print(f"Đã tạo thành công: {filepath} ({count} sensors)")

if __name__ == "__main__":
    main()