from posenet.base_model import BaseModel
import posenet
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PoseNet:

    def __init__(self, model: BaseModel, min_score=0.15):
        self.model = model
        self.min_score = min_score

    def estimate_multiple_poses(self, image, max_pose_detections=10):
        #heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale = \
        #    self.model.predict(image)

        image_scale=1

        #read data from txt file
        headmap =  []
        offsets = []
        fwd = []
        bwd = []
        with open('onnx_output_data/heatmap_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                headmap.append(myarray)
                #print('myarray {}'.format(myarray))
        with open('onnx_output_data/offsets_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                offsets.append(myarray)
        with open('onnx_output_data/displacement_fwd_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                fwd.append(myarray)
        with open('onnx_output_data/displacement_bwd_result.txt') as f:
            lines=f.readlines()
            for line in lines:
                myarray = np.fromstring(line, dtype=float, sep='\n')
                bwd.append(myarray)

        
        headmap = np.array(headmap)
        offsets = np.array(offsets)
        fwd = np.array(fwd)
        bwd = np.array(bwd)
        #print('heatmap_result shape: {}'.format(headmap.shape))
        #heatmap_result = heatmap_result.reshape(1, 9, 9, 17)
        heatmap_result = np.zeros((1, 9,9,17)) #headmap.reshape(1, 9,9,17)
        offsets_result = np.zeros((1, 9, 9, 34))
        displacement_fwd_result = np.zeros((1, 9, 9, 32))
        displacement_bwd_result = np.zeros((1, 9, 9, 32))
        #print('heatmap_result shape: {}'.format(heatmap_result.shape))
        #print('headmap: {}'.format(headmap))
        x=0
        for c in range(17):
            for b in range(9):
                for a in range(9):
                    heatmap_result[0][a][b][c] = headmap[x][0]
                    x=x+1
        x=0
        for c in range(34):
            for b in range(9):
                for a in range(9):
                    offsets_result[0][a][b][c] = offsets[x][0]
                    x=x+1
        x=0
        for c in range(32):
            for b in range(9):
                for a in range(9):
                    displacement_fwd_result[0][a][b][c] = fwd[x][0]
                    x=x+1
        x=0
        for c in range(32):
            for b in range(9):
                for a in range(9):
                    displacement_bwd_result[0][a][b][c] = bwd[x][0]
                    x=x+1
        #print('heatmap_result: {}'.format(heatmap_result))
        '''
        for c in range(17):
            for b in range(9):
                for a in range(9):
                    print('heatmap_result {}'.format(heatmap_result[0][a][b][c]))
        '''
        '''
        heatmap_result=headmap.numpy().reshape(1, 9, 9, 17)
        offsets_result=offsets.numpy().reshape(1, 9, 9, 34)
        displacement_fwd_result=fwd.numpy().reshape(1, 9, 9, 32)
        displacement_bwd_result=bwd.numpy().reshape(1, 9, 9, 32)
        '''
        
        heatmap_result = np.squeeze(heatmap_result)
        offsets_result = np.squeeze(offsets_result)
        displacement_fwd_result = np.squeeze(displacement_fwd_result)
        displacement_bwd_result = np.squeeze(displacement_bwd_result)
        
        height = heatmap_result.shape[0]
        width = heatmap_result.shape[1]
        offsetsxx = offsets_result.reshape(height, width, 2, -1).swapaxes(2, 3)
        displacements_fwd = displacement_fwd_result.reshape(height, width, 2, -1).swapaxes(2, 3)
        displacements_bwd = displacement_bwd_result.reshape(height, width, 2, -1).swapaxes(2, 3) #(33,33,16,2)
        print('heatmap_result.shape[0]: {}'.format(height))
        print('heatmap_result.shape[1]: {}'.format(width))
        print('displacements_fwd.shape: {}'.format(displacements_fwd.shape))
        print('displacements_fwd.shape: {}'.format(displacements_bwd.shape))

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmap_result,
            offsets_result,
            displacement_fwd_result,
            displacement_bwd_result,
            output_stride=32,
            max_pose_detections=max_pose_detections,
            min_pose_score=self.min_score)

        
        #    heatmap_result.numpy().squeeze(axis=0),
        #    offsets_result.numpy().squeeze(axis=0),
        #    displacement_fwd_result.numpy().squeeze(axis=0),
        #    displacement_bwd_result.numpy().squeeze(axis=0),
        #output_stride=self.model.output_stride,

        '''
        print('output_stride: {}'.format(self.model.output_stride))
        print('heatmap_result shape: {}'.format(heatmap_result.numpy().squeeze(axis=0).shape))
        print('offsets_result shape: {}'.format( offsets_result.numpy().squeeze(axis=0).shape))
        print('displacement_fwd_result shape: {}'.format(displacement_fwd_result.numpy().squeeze(axis=0).shape))
        print('displacement_bwd_result shape: {}'.format(displacement_bwd_result.numpy().squeeze(axis=0).shape))
        
        print('heatmap_result shape[0]: {}'.format(heatmap_result.shape[0]))
        pose_scores = np.zeros((10,3))
        print('pose_scores shape: {}'.format(pose_scores.shape))
        print('pose_scores: {}'.format(pose_scores))

        kp_scores = pose_scores[:, 0].copy()
        print('kp_scores shape: {}'.format(kp_scores.shape))
        print('kp_scores: {}'.format(kp_scores))
        '''
        # python image_demo.py --model resnet50 --stride 16 --image_dir ./images --output_dir ./output
        
        #write data to txt file
        f1= open("onnx_output_data/heatmap_result.txt","w+")
        f2= open("onnx_output_data/offsets_result.txt","w+")
        f3= open("onnx_output_data/displacement_fwd_result.txt","w+")
        f4= open("onnx_output_data/displacement_bwd_result.txt","w+")
        #heatmap = np.squeeze(heatmap_result)
        for c in range(17):
            for b in range(9):
                for a in range(9):
                    f1.write('{}\n'.format(heatmap_result[a][b][c]))
        #offsets = np.squeeze(offsets_result)
        for c in range(34):
            for b in range(9):
                for a in range(9):
                    f2.write('{}\n'.format(offsets_result[a][b][c]))
        #displacement_fwd = np.squeeze(displacement_fwd_result)
        for c in range(32):
            for b in range(9):
                for a in range(9):
                    f3.write('{}\n'.format(displacement_fwd_result[a][b][c]))
        #displacement_bwd = np.squeeze(displacement_bwd_result)
        for c in range(32):
            for b in range(9):
                for a in range(9):
                    f4.write('{}\n'.format(displacement_bwd_result[a][b][c]))
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        

        keypoint_coords *= image_scale

        return pose_scores, keypoint_scores, keypoint_coords

    def estimate_single_pose(self, image):
        return self.estimate_multiple_poses(image, max_pose_detections=1)

    def draw_poses(self, image, pose_scores, keypoint_scores, keypoint_coords):
        draw_image = posenet.draw_skel_and_kp(
            image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=self.min_score, min_part_score=self.min_score)

        return draw_image

    def print_scores(self, image_name, pose_scores, keypoint_scores, keypoint_coords):
        print()
        print("Results for image: %s" % image_name)
        f5= open("output_data/image_name.txt","w+")
        f5.write('{}\n'.format(image_name))
        f5.close()
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
