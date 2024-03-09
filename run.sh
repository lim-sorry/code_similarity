python -u main.py --model_path neulab/codebert-cpp --model_name codebert-cpp --pt_path codebert-cpp.pt --mode train --batch_size 16
python -u main.py --model_path neulab/codebert-cpp --model_name codebert-cpp --pt_path codebert-cpp.pt --mode test --batch_size 256
python -u main.py --model_path MickyMike/codebert-c --model_name codebert-c --pt_path codebert-c.pt --mode train --batch_size 16
python -u main.py --model_path MickyMike/codebert-c --model_name codebert-c --pt_path codebert-c.pt --mode test --batch_size 256

