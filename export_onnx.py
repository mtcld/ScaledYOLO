import argparse

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Mish
from onnxsim import simplify

import tensorrt as trt

def save_trt_engine(engine, path):
    """Serialize TensorRT engine to disk.
    Arguments:
        engine (tensorrt.ICudaEngine): TensorRT engine to serialize
        path (str): disk path to write the engine
    """
    with open(path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))

# build tensorRT engine from onnx_file_path and save at engine_file_path
def build_engine(onnx_file_path,engine_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
        print('use FP16 mode inference !!')

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    print("Completed creating Engine")
    print('Saving engine ...')
    save_trt_engine(engine,engine_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov4-p5.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[1024, 1024], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--engine_file',type=str, default=None, help='output engine file path')
    parser.add_argument('--model_data_file',type=str, default=None, help='output model data (grid,stride,...) file path')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model

    # Update model
    
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
        if isinstance(m, models.common.Conv) and isinstance(m.act, models.common.Mish):
            m.act = Mish()  # assign activation
        if isinstance(m, models.common.BottleneckCSP) or isinstance(m, models.common.BottleneckCSP2) \
                or isinstance(m, models.common.SPPCSP):
            if isinstance(m.bn, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(m.bn.num_features, eps=m.bn.eps, momentum=m.bn.momentum)
                bn.training = False
                bn._buffers = m.bn._buffers
                bn._non_persistent_buffers_set = set()
                m.bn = bn
            if isinstance(m.act, models.common.Mish):
                m.act = Mish()  # assign activation
    
        # if isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)

    model.eval()

    # Save spec of model
    detector = model.model[-1]
    db = {
        'num_detector_layer': detector.nl,
        'grid': detector.grid,
        'anchor_grid': detector.anchor_grid,
        'stride' : detector.stride
    }
    print('Saving spec of model into ',opt.model_data_file,'....')
    torch.save(bd,opt.model_data_file)

    model.model[-1].export = True  # set Detect() layer export=True
    # y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)

    except Exception as e:
        print('ONNX export failure: %s' % e)

    # tensorRT build engine and save in .trt file
    try :
        build_engine(f,opt.engine_file)
    except Exception as e:
        print('TensorRT export failure: %s' % e)

