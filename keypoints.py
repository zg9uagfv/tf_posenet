#export type Tuple<T> = [T, T];
#export type StringTuple = Tuple<string>;
#export type NumberTuple = Tuple<number>;

partNames = [
    'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
    'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
    'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
]

NUM_KEYPOINTS = len(partNames)

'''
export interface NumberDict {
  [jointName: string]: number;
}
'''

partIds = {}
def analyse_part_names(part_names, dict_part_ids):
    for i in range(NUM_KEYPOINTS):
        dict_part_ids[part_names[i]] = i
analyse_part_names(partNames, partIds)

'''
export const partIds =
    partNames.reduce((result: NumberDict, jointName, i): NumberDict => {
      result[jointName] = i;
      return result;
    }, {}) as NumberDict;
'''

connectedPartNames = [
  ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
  ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
  ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
  ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
  ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
];

'''
 * Define the skeleton. This defines the parent->child relationships of our
 * tree. Arbitrarily this defines the nose as the root of the tree, however
 * since we will infer the displacement for both parent->child and
 * child->parent, we can define the tree root as any node.
'''
poseChain= [
  ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
  ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
  ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
  ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
  ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
  ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
  ['rightKnee', 'rightAnkle']
]

'''
export const connectedPartIndices = connectedPartNames.map(
    ([jointNameA, jointNameB]) => ([partIds[jointNameA], partIds[jointNameB]]));
'''

connectedPartIndices = []

def analyse_connected_part_indices(connected_part_names, dict_part_ids, \
                                   connected_part_indices):
    for jointNameA, jointNameB in connected_part_names:
        connected_part_indices.append([dict_part_ids[jointNameA], \
                                     dict_part_ids[jointNameB]])

analyse_connected_part_indices(connectedPartNames, partIds, connectedPartIndices)

partChannels = [
    'left_face',
    'right_face',
    'right_upper_leg_front',
    'right_lower_leg_back',
    'right_upper_leg_back',
    'left_lower_leg_front',
    'left_upper_leg_front',
    'left_upper_leg_back',
    'left_lower_leg_back',
    'right_feet',
    'right_lower_leg_front',
    'left_feet',
    'torso_front',
    'torso_back',
    'right_upper_arm_front',
    'right_upper_arm_back',
    'right_lower_arm_back',
    'left_lower_arm_front',
    'left_upper_arm_front',
    'left_upper_arm_back',
    'left_lower_arm_back',
    'right_hand',
    'right_lower_arm_front',
    'left_hand'
]