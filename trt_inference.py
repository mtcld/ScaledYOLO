import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
from utils.datasets import LoadImagesBatch
import math
from collections import OrderedDict
from utils.general import (check_img_size, non_max_suppression, scale_coords)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRT_model():
    def __init__(self,engine_file_path,model_data_file_path,confident_score,iou_threshold,label):
        self.engine, self.context = self.load_trt_engine(engine_file_path)
        db = torch.load(model_data_file_path)
        self.num_detector_layer = db['num_detector_layer']
        self.grid = db['grid']
        self.anchor_grid = db['anchor_grid']
        self.stride = db['stride']
        self.no = 6 # number of class + 5 = number of output per anchor

        self.conf_score = confident_score
        self.iou_thres = iou_threshold
        self.label = label

    def load_trt_engine(self,path):
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

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference_trt(self, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]
    
    def pre_process(self,input):
        pass

    def post_process(self, output):
        pass

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def inference(self,img_paths):
        dataset = LoadImagesBatch(img_paths,img_size=896)

        batch = []
        shapes = []
        paths = []
        for path, img, im0s, shape in dataset:
            #img = torch.from_numpy(img).to(self.device)
            #img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = torch.from_numpy(img).float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            batch.append(img)
            shapes.append(shape)
            paths.append(path)
        self.batch_size = len(paths)
        batch = torch.cat(batch)
        batch = np.array(batch,dtype=np.float32,order='C')

        print('batch shape :',batch.shape)

        ## inference tensorRT
        inputs, outputs, bindings, stream = self.allocate_buffers()
        inputs[0].host = batch
        trt_outputs = self.do_inference_trt(bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        print('done : ',len(trt_outputs) == self.num_detector_layer)
        raw_output = [] 
        for i in range(self.num_detector_layer):
            feature_size = math.sqrt(len(trt_outputs[i])/self.batch_size/4/self.no)
            feature_size = int(feature_size)
            y = torch.Tensor(trt_outputs[i]).reshape(self.batch_size,4,feature_size,feature_size,self.no)

            if self.grid[i].shape[2:4] != y.shape[2:4]:
                self.grid[i] = self._make_grid(feature_size,feature_size)
            
            y = y.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            raw_output.append(y.view(self.batch_size, -1, self.no))
        
        raw_output = torch.cat(raw_output,1)

        raw_output = non_max_suppression(raw_output, self.conf_score, self.iou_thres, agnostic=True)

        output = OrderedDict()
        for i, det in enumerate(raw_output):  # detections per image
            out_boxes = OrderedDict()
            boxes = []
            labels = []
            scores = []
            if det is not None and len(det):
                # remove batch padding and rescale each image to its original size
                det[:, :4] = scale_coords(batch.shape[2:], det[:, :4], shapes[i][0],shapes[i][1]).round()
               
                # convert tensor to list and scalar
                for *xyxy, conf, cls in det:
                    rec = torch.tensor(xyxy).view(1,4).view(-1).int().tolist()
                    conf = (conf).view(-1).detach().tolist()[0]
                    h,w = shapes[i][0]
                    boxes.append(rec)
                    labels.append(self.label)
                    scores.append(conf)

            out_boxes['boxes'] = boxes
            out_boxes['labels'] = labels
            out_boxes['scores'] = scores   
            output[paths[i]] = out_boxes

        return output

def main():
    
    paths = ['scratch_yolo/valid/https:__s3.amazonaws.com_mc-imt_vehicle_2020D3862_detail_damage2_51519_medium_A902874A-D6F3-45F2-994F-98A756E21547.jpeg']
    engine_file_path = 'scratch.trt'
    model_data_file_path = 'scratch_db'
    model = TensorRT_model(engine_file_path,model_data_file_path,0.3,0.3,'scratch')
    out = model.inference(paths)

    img = cv2.imread(paths[0])
    for box in out[paths[0]]['boxes']:
        print(box)
        img = cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255, 0, 0),1)

    cv2.imwrite('test.png',img)

if __name__ == '__main__':
    main()
