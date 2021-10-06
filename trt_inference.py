import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2

ONNX_FILE_PATH = "checkpoints/scratch/scratch_23_9.onnx"
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def save_trt_engine(engine, path):
    """Serialize TensorRT engine to disk.
    Arguments:
        engine (tensorrt.ICudaEngine): TensorRT engine to serialize
        path (str): disk path to write the engine
    """
    with open(path, mode='wb') as f:
        f.write(bytearray(engine.serialize()))

def load_trt_engine(path):
    """Deserialize TensorRT engine from disk.
    Arguments:
        path (str): disk path to read the engine
    Returns:
        tensorrt.ICudaEngine: the TensorRT engine loaded from disk
    """
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine, engine.create_execution_context()

def build_engine(onnx_file_path):
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

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    save_trt_engine(engine,'scratch.trt')
    context = engine.create_execution_context()
    #print("Completed creating Engine")
    return engine, context

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def main():
    # initialize TensorRT engine and parse ONNX model
    #engine, context = build_engine(ONNX_FILE_PATH)
    engine, context = load_trt_engine('scratch.trt')
    
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    print('input : ',inputs)
    print(len(outputs[0].host))


    # preprocess input data
    #host_input = np.array(preprocess_image("turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    image = cv2.imread('scratch_yolo/valid/https:__s3.amazonaws.com_mc-ai_dataset_india_20190312_1_IMG_20170417_101024.jpg')
    image = cv2.resize(image, (896,896), interpolation = cv2.INTER_AREA)
    image = image / 255.0
    image = image[:, :, ::-1].transpose(2, 0, 1)    
    image = np.expand_dims(image, axis=0)
    image = np.array(image,dtype=np.float32,order='C')
    print(image.shape)
    inputs[0].host = image

    trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print('done : ',len(trt_outputs))

    # postprocess results
    #print(host_output)
    #output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])


if __name__ == '__main__':
    main()
