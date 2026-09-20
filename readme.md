# tự tạo thư mục và cài thư viện vô nha 


# chú ý do là chạy trên kaggle lên ta sẽ ko có dữ liệu đầu vào thay vào đó ta sửa file runner để truyền dữu liệu từ folder data tự tạo 

chạy tất cả bộ dữ liệu 
python main.py

chạy bộ mình chỉ định 
python main.py sample30.json

python main.py nodes_150.json

python main.py nodes_250.json

chạy vẽ biểu đồ ( lưu ý phải chạy chương trình trước )
python plot_results.py --folder results/folder_sample30

python plot_results.py --folder results/folder_nodes_150

python plot_results.py --folder results/folder_nodes_250

python plot_results.py --folder results/folder_nodes_35

# Chạy file để tự động sinh dữ liệu
python auto_convert.py

# Muốn cắm máy chạy toàn bộ
python batch_runner.py

# Chỉ chạy thư mục Exponential (copy đúng tên thư mục dán vào)
python batch_runner.py Exponential_Distribution


Chỉ test thử duy nhất 1 file (ví dụ file 150_1)
python batch_runner.py 150_1

python batch_runner.py Exponential_Distribution/150_1
python batch_runner.py Normal_Distribution/150_1
python batch_runner.py Uniform_Distribution/150_1
