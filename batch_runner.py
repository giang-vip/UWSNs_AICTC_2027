import os
import sys
import glob
from uwsn import runner

INPUT_BASE = "Kmean_Data"
OUTPUT_BASE = "Results"

def run_batch(target):
    # 1. Tìm kiếm dữ liệu dựa vào lệnh bạn gõ
    if target == "": 
        # Không gõ gì -> Quét toàn bộ Kmean_Data
        search_path = os.path.join(INPUT_BASE, '**', '*.json')
        json_files = glob.glob(search_path, recursive=True)
    elif "Distribution" in target and ("/" in target or "\\" in target):
        # Trường hợp gõ cụ thể kèm phân bố (VD: Exponential_Distribution/150_1)
        folder_part, file_part = os.path.split(target)
        search_path = os.path.join(INPUT_BASE, folder_part, '**', f'{file_part}.json')
        json_files = glob.glob(search_path, recursive=True)
        # Nếu tìm theo cách này không ra (do người dùng gõ thiếu đuôi .json), ta thử lọc chính xác tên file
        if not json_files:
            search_path = os.path.join(INPUT_BASE, folder_part, '**', '*.json')
            all_files = glob.glob(search_path, recursive=True)
            json_files = [f for f in all_files if os.path.basename(f) == f"{file_part}.json"]
    elif "Distribution" in target: 
        # Chỉ gõ tên thư mục phân bố -> Quét toàn bộ thư mục đó
        search_path = os.path.join(INPUT_BASE, target, '**', '*.json')
        json_files = glob.glob(search_path, recursive=True)
    else: 
        # Trường hợp gõ tên file đơn thuần (VD: 150_1) -> Khớp tuyệt đối tên file để không bị dính sang 150_10, 150_11...
        search_path = os.path.join(INPUT_BASE, '**', '*.json')
        all_files = glob.glob(search_path, recursive=True)
        json_files = [f for f in all_files if os.path.basename(f) == f"{target}.json"]

    print(f"🚀 Tìm thấy {len(json_files)} kịch bản cần chạy.\n")

    if len(json_files) == 0:
        print(f"Không tìm thấy kịch bản nào phù hợp với từ khóa: '{target}'")
        return

    # 2. Vòng lặp chạy mô phỏng
    for json_file in json_files:
        rel_path = os.path.relpath(json_file, INPUT_BASE)
        dir_name = os.path.dirname(rel_path).replace("\\json", "").replace("/json", "")
        base_name = os.path.basename(json_file).replace('.json', '')
        
        output_dir = os.path.join(OUTPUT_BASE, dir_name, f"folder_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"--- Đang chạy kịch bản: {base_name} ---")
        runner.main(input_file=json_file, output_folder=output_dir)

if __name__ == "__main__":
    # Lấy tham số truyền từ Terminal
    user_input = sys.argv[1] if len(sys.argv) > 1 else ""
    run_batch(user_input)