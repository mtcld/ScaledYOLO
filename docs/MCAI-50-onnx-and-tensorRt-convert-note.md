# Convert .pt yolo model to tensorRT .trt engine file
- Use [export_onnx.py](../export_onnx.py) to convert
- input : 
    - .pt yolo weight file path
    - image_size (default 1024)
    - batch_size (default 1)
    - .trt output engine file
    - model spec store file (file store tride, number of detection layer, grid, anchor grid in tensor format)
- Example :
```
python export_onnx.py --weights scratch.pt --engine_file scratch.trt --model_data_file scrath_db
```
