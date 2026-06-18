# ViDEC: Mô phỏng kiểm tra khuyết tật dưới nước

ViDEC là prototype pipeline kiểm tra khuyết tật bề mặt công trình dưới nước.

Pipeline hiện tại dùng HoloOcean/Unreal Engine để mô phỏng AUV, camera RGB và metadata runtime; dùng YOLO11s-seg để phát hiện/phân vùng vùng nghi ngờ khuyết tật; và tạo evidence packet gồm full frame, ROI crop, mask+bbox, metadata compact cho bài toán truyền thông giới hạn băng thông.

Runtime YOLO là evidence định tính nếu chưa có ground-truth mask align trực tiếp với camera runtime.

## Repository Layout

```text
ViDEC_Project/
├── configs/
│   ├── class_mapping.yaml                         # Mapping tên class/ID class dùng chung trong dự án
│   ├── holoocean/
│   │   ├── config.json                            # Khai báo custom HoloOcean world tên ViDEC
│   │   ├── test_ocean_map-HoveringCamera.json     # Scenario HoloOcean: world, AUV, sensor, camera, pose khởi tạo
│   │   └── auv_keyboard_teleop.yaml               # Cấu hình điều khiển AUV bằng bàn phím và capture frame runtime
│   └── yolo/
│       └── s2ds_seg4.yaml                         # Config dataset YOLO segmentation 4 class
│
├── scripts/
│   ├── holoocean/
│   │   ├── capture_videc_holoocean_runtime.py     # Capture frame và metadata từ HoloOcean runtime
│   │   ├── diagnose_auv_action_space.py           # Kiểm tra action space/thruster control của AUV
│   │   ├── generate_evidence_packets.py           
│   │   ├── make_auv_teleop_video.py               
│   │   ├── register_videc_holoocean_world.py      # Copy/đăng ký config world ViDEC vào thư mục HoloOcean local
│   │   ├── teleop_auv_keyboard_capture.py         # Điều khiển AUV bằng bàn phím, capture RGB frame và metadata
│   │   ├── test_videc_holoocean_make.py           # Test holoocean.make() với scenario ViDEC
│   │   └── update_videc_scenario_pose.py          # Cập nhật pose/góc quay khởi tạo của AUV trong scenario JSON
│   │
│   ├── model/
│   │   ├── evaluate.py                            # Script đánh giá/phân tích kết quả mô hình
│   │   ├── predict_yolo_runtime_inspection.py     # Chạy YOLO11s-seg trên ảnh runtime, lưu JSON/overlay/mask/mask+bbox
│   │   ├── train_yolo11s_seg_s2ds.py              # Huấn luyện YOLO11s-seg trên dataset S2DS đã convert
│   │   └── val_yolo11s_seg_s2ds.py                # Validate/test YOLO11s-seg trên validation hoặc test split
│   │
│   ├── evidence/                                  # Visualize evidence packet và tạo report size comparison
│   │   ├── build_inspection_evidence_packets.py   
│   │   ├── merge_yolo_predictions_with_runtime_metadata.py         
│   │   ├── report_size_comparison.py              
│   │   ├── visualize_inspection_evidence_packet.py                           
│   │   └── visualize_yolo_inspection_evidence.py  
│   │
│   └── process_data/
│       ├── augment_s2ds_underwater.py             # Tạo underwater augmentation cho ảnh/mask S2DS
│       ├── convert_s2ds_to_yolo_seg.py            # Chuyển mask màu S2DS sang YOLO segmentation polygon format
│       ├── inspect_s2ds.py                        # Kiểm tra số lượng/cấu trúc ảnh và mask trong S2DS gốc
│       ├── prepare_s2ds_subset.py                 # Tạo subset S2DS nhỏ để test nhanh pipeline
│       └── visualize_yolo_seg_labels.py           # Visualize polygon/bbox sau khi convert sang YOLO format
│
├── data/                                          # Lưu data
├── reports/                                       
├── requirements.txt                               
├── README.md                                      
└── .gitignore                                     
```


Generated data, model weights, HoloOcean packaged Linux builds, training runs, old phase docs, and report outputs are not stored in Git.

## Môi trường đã kiểm thử

```text
Ubuntu 22.04
Python 3.10
PyTorch 2.5.1+cu121
Unreal Engine 5.3.2
HoloOcean 2.3.0
GPU: NVIDIA RTX 3060 12GB
```

## Cài đặt

Tạo virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Nếu cần CUDA PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Nếu chạy YOLO:

```bash
pip install ultralytics
```
## Cài đặt phần mềm phụ thuộc

### Cài Unreal Engine 5.3.2
ViDEC sử dụng custom world trong HoloOcean/Unreal Engine, vì vậy cần cài Unreal Engine trước khi build hoặc chỉnh sửa môi trường mô phỏng.

Link Unreal Engine:

https://www.unrealengine.com/download

Link hướng dẫn kết nối Epic Games với GitHub:

https://www.unrealengine.com/en-US/ue-on-github

Các bước tổng quát:

Tạo hoặc đăng nhập tài khoản Epic Games.
Kết nối tài khoản Epic Games với GitHub.
Chấp nhận Unreal Engine EULA.
Cài Unreal Engine 5.3.2.
Kiểm tra Unreal Editor có thể mở project HoloOcean/Holodeck.

Lưu ý: project này được build với Unreal Engine 5.3.2. Nên dùng đúng version này để tránh lỗi không tương thích khi package world.

### Cài HoloOcean 2.3.0

Trong setup này, HoloOcean được cài từ source vì cần làm việc với Unreal Engine và custom world.

Tài liệu HoloOcean:

https://byu-holoocean.github.io/holoocean-docs/

Tài liệu HoloOcean UE5.3:

https://byu-holoocean.github.io/holoocean-docs/UE5.3_Prerelease/index.html

Các bước tổng quát:

Đảm bảo tài khoản GitHub đã được liên kết với Epic Games.
Clone source HoloOcean/Holodeck tương ứng với Unreal Engine 5.3.
Tạo môi trường Python.
Cài Python client của HoloOcean từ source.

```text
git clone https://github.com/byu-holoocean/HoloOcean.git holoocean
cd holoocean
pip install -e .
```

Kiểm tra cài đặt:
```text
python -c "import holoocean; print(holoocean.__version__)"
```
Nếu import được holoocean và version là 2.3.0, môi trường Python đã nhận HoloOcean.

### Chuẩn bị HoloOcean world

Thư mục world của HoloOcean

Sau khi cài HoloOcean, các world local nằm trong:

~/.local/share/holoocean/2.3.0/worlds/

Custom world của project này dùng tên: ViDEC

Cấu trúc mong muốn:
```text
~/.local/share/holoocean/2.3.0/worlds/ViDEC/

├── config.json
├── test_ocean_map-HoveringCamera.json
└── Linux/
```
Trong đó:
``` text
config.json                         # Khai báo custom world ViDEC
test_ocean_map-HoveringCamera.json  # Scenario spawn AUV, camera và sensor
Linux/                              # Packaged build từ Unreal/Holodeck
```
Repo chỉ lưu các file cấu hình nhỏ tại:
```text
configs/holoocean/
├── config.json
├── test_ocean_map-HoveringCamera.json
└── auv_keyboard_teleop.yaml
```
Copy config vào HoloOcean world local:
```text
mkdir -p ~/.local/share/holoocean/2.3.0/worlds/ViDEC

cp configs/holoocean/config.json \
  ~/.local/share/holoocean/2.3.0/worlds/ViDEC/

cp configs/holoocean/test_ocean_map-HoveringCamera.json \
  ~/.local/share/holoocean/2.3.0/worlds/ViDEC/
```

### Copy packaged Linux build

Packaged ViDEC world: https://drive.google.com/file/d/1rKrLyESxGIB1ef2SbLZW_PrEh4EARGir/view?usp=sharing

Sau khi tải về, giải nén vào thư mục world của HoloOcean: 
```text
mkdir -p ~/.local/share/holoocean/2.3.0/worlds
unzip ViDEC_world_holoocean_2.3.0_linux.zip -d ~/.local/share/holoocean/2.3.0/worlds/
```

Mỗi khi thay đổi package hoặc chỉnh sửa map trong Unreal, cần sync lại build mới vào world ViDEC:
```text
pkill -f Holodeck

rm -rf ~/.local/share/holoocean/2.3.0/worlds/ViDEC/Linux

cp -a ~/coding/holoocean/dist_videc/Linux \
  ~/.local/share/holoocean/2.3.0/worlds/ViDEC/Linux
```
Sau khi copy xong, kiểm tra:
```text
ls ~/.local/share/holoocean/2.3.0/worlds/ViDEC/
```
Kết quả cần có:

config.json
test_ocean_map-HoveringCamera.json
Linux/

Lưu ý:

- Packaged Linux build không có trong Git.
- Không ghi đè world `Ocean` mặc định.
- Sau khi package lại Unreal, cần sync build mới vào `~/.local/share/holoocean/2.3.0/worlds/ViDEC/Linux/`.

## Chuẩn bị model

Đặt model fine-tuned tại:

```text
weights/best.pt
```

hoặc truyền path trực tiếp bằng `--model`.

## Chuẩn bị data

Dataset: https://github.com/ben-z-original/s2ds

Sau khi tải dataset, đặt vào:

```text
data/raw/s2ds/
```

Output data sẽ sinh ở:

```text
data/augmented/
data/yolo_s2ds_seg4/
```

## Main Usage

### 1. Test HoloOcean scenario

```bash
python scripts/holoocean/test_videc_holoocean_make.py \
  --scenario test_ocean_map-HoveringCamera
```

Expected: `holoocean.make()` start thành công và state có các key như `PoseSensor`, `DepthSensor`, `LeftCamera`, `RightCamera`, và `t`.

### 2. Teleop AUV và capture runtime frames

```bash
python scripts/holoocean/teleop_auv_keyboard_capture.py \
  --config configs/holoocean/auv_keyboard_teleop.yaml \
  --preview
```

Output:

```text
data/runtime_teleop/frames/
data/runtime_teleop/metadata/
```

AUV được spawn bởi HoloOcean scenario. Không cần đặt AUV thủ công trong Unreal map nếu scenario đã spawn AUV.

### 3. Tạo video từ teleop frames

```bash
python scripts/holoocean/make_auv_teleop_video.py \
  --frames data/runtime_teleop/frames \
  --metadata-dir data/runtime_teleop/metadata \
  --output reports/figures/auv_keyboard_teleop.mp4 \
  --auto-fps
```

### 4. Chạy YOLO runtime inference

```bash
python scripts/model/predict_yolo_runtime_inspection.py \
  --model weights/best.pt \
  --source data/runtime_teleop/frames \
  --output data/yolo_runtime_predictions \
  --conf 0.35 \
  --device 0
```

Output:

```text
data/yolo_runtime_predictions/predictions/
data/yolo_runtime_predictions/overlays/
data/yolo_runtime_predictions/masks/
data/yolo_runtime_predictions/mask_bbox/
```

### 5. Build evidence packets

```bash
python scripts/evidence/build_inspection_evidence_packets.py \
  --predictions data/yolo_runtime_predictions/predictions \
  --overlays data/yolo_runtime_predictions/overlays \
  --mask-bbox data/yolo_runtime_predictions/mask_bbox \
  --frames data/runtime_teleop/frames \
  --metadata data/runtime_teleop/metadata \
  --output data/inspection_evidence_packets
```

Output:

```text
data/inspection_evidence_packets/packets/
data/inspection_evidence_packets/roi/
data/inspection_evidence_packets/metadata_compact/
data/inspection_evidence_packets/size_comparison/
reports/results/inspection_evidence_packet_summary.csv
```

### 6. Visualize evidence packet

```bash
python scripts/evidence/visualize_inspection_evidence_packet.py \
  --packets data/inspection_evidence_packets/packets \
  --output reports/figures/inspection_evidence_packet_4panel.png \
  --font-size 24 \
  --make-grid
```

Một sample cụ thể:

```bash
python scripts/evidence/visualize_inspection_evidence_packet.py \
  --packets data/inspection_evidence_packets/packets \
  --frame-id frame_000059 \
  --output reports/figures/inspection_evidence_packet_frame_000059.png \
  --font-size 24
```

### 7. Report size comparison

```bash
python scripts/evidence/report_size_comparison.py \
  --packets data/inspection_evidence_packets/packets \
  --output-csv reports/results/inspection_size_comparison.csv \
  --output-json reports/results/inspection_size_comparison.json \
  --output-figure reports/figures/inspection_size_comparison.png
```

## Optional Data/Training

Nếu cần rebuild YOLO segmentation dataset từ S2DS:

```bash
python scripts/process_data/augment_s2ds_underwater.py \
  --input-root data/raw/s2ds \
  --output-root data/augmented/s2ds_underwater \
  --num-aug-per-image 2 \
  --include-original
```

```bash
python scripts/process_data/convert_s2ds_to_yolo_seg.py \
  --input-root data/augmented/s2ds_underwater \
  --output-root data/yolo_s2ds_seg4 \
  --include-hard-negative
```

```bash
python scripts/process_data/visualize_yolo_seg_labels.py \
  --dataset data/yolo_s2ds_seg4 \
  --yaml configs/yolo/s2ds_seg4.yaml \
  --split train \
  --samples 24
```

Train YOLO11s-seg:

```bash
python scripts/model/train_yolo11s_seg_s2ds.py \
  --data configs/yolo/s2ds_seg4.yaml \
  --model yolo11s-seg.pt \
  --epochs 50 \
  --batch 8 \
  --device 0
```

Validate:

```bash
python scripts/model/val_yolo11s_seg_s2ds.py \
  --model runs/videc_yolo/yolo11s_seg_s2ds4_underwater_v1/weights/best.pt \
  --data configs/yolo/s2ds_seg4.yaml \
  --split test
```

## Output folders

Output chính:

```text
data/runtime_teleop/
data/yolo_runtime_predictions/
data/inspection_evidence_packets/
reports/results/
reports/figures/
```

