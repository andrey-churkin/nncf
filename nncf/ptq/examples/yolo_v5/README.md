# Quantize the Ultralytics YOLOv5 model and check accuracy

To work with this sample you should do the following steps:

1. Clone the `yolov5` repository into `${NNCF_ROOT}/nncf/ptq/examples/yolo_v5` folder and install requirements.
   To do this run the following command:

   ```bash
   NNCF_ROOT="" # absolute path to the NNCF repository
   cd ${NNCF_ROOT}/nncf/ptq/examples/yolo_v5
   git clone https://github.com/ultralytics/yolov5.git -b v6.2
   pip install -r yolov5/requirements.txt
   sudo apt install unzip
   ```

2. Set the `PYTHONPATH` environment variable value to be the path to the `yolov5` directory.
   To do this run the following command:

   ```bash
   export PYTHONPATH=${PYTHONPATH}:${NNCF_ROOT}/nncf/ptq/examples/yolo_v5/yolov5
   ```

3. Run the example:

   ```bash
   cd ${NNCF_ROOT}/nncf/ptq/examples/yolo_v5
   python yolo_v5_quantization.py
   ```

   You do not need to prepare the COCO 2017 validation dataset. It will be downloaded automatically:

   ```
   yolo_v5
   ├── yolov5
   └── datasets
       └── coco  <- Here
   ```
