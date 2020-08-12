from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class BaseModel(ABC):

    # keys for the output_tensor_names map
    HEATMAP_KEY = "heatmap"
    OFFSETS_KEY = "offsets"
    DISPLACEMENT_FWD_KEY = "displacement_fwd"
    DISPLACEMENT_BWD_KEY = "displacement_bwd"

    def __init__(self, model_function, output_tensor_names, output_stride):
        self.output_stride = output_stride
        self.output_tensor_names = output_tensor_names
        self.model_function = model_function

    def valid_resolution(self, width, height):
        # calculate closest smaller width and height that is divisible by the stride after subtracting 1 (for the bias?)
        target_width = (int(width) // self.output_stride) * self.output_stride + 1
        target_height = (int(height) // self.output_stride) * self.output_stride + 1
        return target_width, target_height

    @abstractmethod
    def preprocess_input(self, image):
        pass

    def predict(self, image):
        input_image, image_scale = self.preprocess_input(image)

        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

        result = self.model_function(input_image)
        #print("result: {}".format(result))
        
        heatmap_result = result[self.output_tensor_names[self.HEATMAP_KEY]]
        offsets_result = result[self.output_tensor_names[self.OFFSETS_KEY]]
        displacement_fwd_result = result[self.output_tensor_names[self.DISPLACEMENT_FWD_KEY]]
        displacement_bwd_result = result[self.output_tensor_names[self.DISPLACEMENT_BWD_KEY]]
        
        '''
        f1= open("output/heatmap_result.txt","r")
        f2= open("output/offsets_result.txt","r")
        f3= open("output/displacement_fwd_result.txt","r")
        f4= open("output/displacement_bwd_result.txt","r")
        #print("heatmap_result: {}".format(f1.readline()))
        #print("heatmap_result: {}".format(f1.readline()))
        
        for c in range(17):
            for b in range(33):
                for a in range(33):
                    heatmap_result[a][b][c] = f1.readline()
        for c in range(34):
            for b in range(33):
                for a in range(33):
                    offsets_result[a][b][c] = f2.readline()
        for c in range(32):
            for b in range(33):
                for a in range(33):
                    displacement_fwd_result[a][b][c] = f3.readline()
        for c in range(32):
            for b in range(33):
                for a in range(33):
                    displacement_bwd_result[a][b][c] = f4.readline()
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        '''
        '''
        headmap =  []
        offsets = []
        fwd = []
        bwd = []
        with open('output/heatmap_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                headmap.append(myarray)
        with open('output/offsets_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                offsets.append(myarray)
        with open('output/displacement_fwd_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                fwd.append(myarray)
        with open('output/displacement_bwd_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                bwd.append(myarray)
        heatmap_result = np.array(headmap)
        offsets_result = np.array(offsets)
        displacement_fwd_result = np.array(fwd)
        displacement_bwd_result = np.array(bwd)
        heatmap_result = heatmap_result.reshape(1, 9, 9, 17)
        offsets_result = offsets_result.reshape(1, 9, 9, 34)
        displacement_fwd_result = displacement_fwd_result.reshape(1, 9, 9, 32)
        displacement_bwd_result = displacement_bwd_result.reshape(1, 9, 9, 32)
        
        '''

        return tf.sigmoid(heatmap_result), offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale
