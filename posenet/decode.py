import numpy as np

from posenet.constants import *


def traverse_to_targ_keypoint(
        edge_id, source_keypoint, target_keypoint_id, scores, offsets, output_stride, displacements
):
    height = scores.shape[0]
    width = scores.shape[1]

    source_keypoint_indices = np.clip(
        np.round(source_keypoint / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    #print('source_keypoint_indices: {}'.format(source_keypoint_indices))

    displaced_point = source_keypoint + displacements[
        source_keypoint_indices[0], source_keypoint_indices[1], edge_id]

    displaced_point_indices = np.clip(
        np.round(displaced_point / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)
    #print('displaced_point_indices: {}'.format(displaced_point_indices))

    score = scores[displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    image_coord = displaced_point_indices * output_stride + offsets[
        displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    #print('score: {}'.format(score))
    #print('image_coord: {}'.format(image_coord))
    return score, image_coord


def decode_pose(
        root_score, root_id, root_image_coord,
        scores,
        offsets,
        output_stride,
        displacements_fwd,
        displacements_bwd
):
    num_parts = scores.shape[2]
    num_edges = len(PARENT_CHILD_TUPLES)

    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord

    for edge in reversed(range(num_edges)): # 15 -> 0
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
        #print('edge: {}'.format(edge))
        #print('source_keypoint_id: {}'.format(source_keypoint_id))
        #print('target_keypoint_id: {}'.format(target_keypoint_id))
        #print('instance_keypoint_scores[source_keypoint_id]: {}'.format(instance_keypoint_scores[source_keypoint_id]))
        #print('instance_keypoint_scores[target_keypoint_id]: {}'.format(instance_keypoint_scores[target_keypoint_id]))
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_bwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    for edge in range(num_edges): # 0 -> 15
        source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
        #print('edge: {}'.format(edge))
        #print('source_keypoint_id: {}'.format(source_keypoint_id))
        #print('target_keypoint_id: {}'.format(target_keypoint_id))
        #print('instance_keypoint_scores[source_keypoint_id]: {}'.format(instance_keypoint_scores[source_keypoint_id]))
        #print('instance_keypoint_scores[target_keypoint_id]: {}'.format(instance_keypoint_scores[target_keypoint_id]))
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_fwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    return instance_keypoint_scores, instance_keypoint_coords
