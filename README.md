# Driver-Face-Identification

## Quick Start

### 1. Install

### 2. Data Generation

```bash
pytohn data_generator.py --save_path SAVE_PATH --cls_name CLS_NAME --size SIZE
```

### 3. Train

```bash
python classifier_train.py --epochs EPOCHS --batch_size BATCH_SIZE --dataset DATA_PATH --num_workers 8
```

### 4. Demo

```bash
python demo.py --model MODEL_PATH --names NAMES_PATH --size SIZE --source VIDEO or WEBCAM --target TARGET_CLS_ID
```

## References

1.timesler / facenet-pytorch: https://github.com/timesler/facenet-pytorch

2.intel-iot-devkit / sample-videos: https://github.com/intel-iot-devkit/sample-videos
