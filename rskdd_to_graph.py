# coding:UTF-8
import os
import json
import string

'''
created by cjf 2021-03-18
create a json file of points pair from RSKDD data
'''

def create_json(input_kp_path, input_desc_path, output_josn_path, pose):
    data = json.load(open('/home/cuijiafeng/home/cuijiafeng/deep_learning/SG_PR/data/0.json'))
    
    kp = open(input_kp_path)  # 特征点路径
    desc = open(input_desc_path) # 描述子路径
    list_of_kp = list(kp)
    list_of_desc = list(desc)
    data['centers'].clear()
    data['nodes'].clear()
    data['pose'].clear()

    saliency = list()
    for i in range(len(list_of_kp)):
        s = list_of_kp[i].split(' ', -1)
        saliency.append(float(s[3]))
    
    sorted_saliency = sorted(saliency)[0:100]
    index = []
    for i in sorted_saliency:
        index.append(saliency.index(i))

    for i in index:
        kps = list_of_kp[i].split(' ', -1)
        kps = kps[0:3]
        descs = list_of_desc[i].split(' ', -1)
        kps[0] = float(kps[0])
        kps[1] = float(kps[1])
        kps[2] = float(kps[2])
        for j in range(len(descs)):
            descs[j] = float(descs[j])
        data['centers'].append(kps)
        data['nodes'].append(descs)

        # print(s)
        # break

    pose_str = pose.split(' ', -1)

    for i in range(len(pose_str)):
        data['pose'].append(float(pose_str[i]))

    with open(output_josn_path, 'w') as file_obj:
        json.dump(data, file_obj)


f_pose = open('/home/cuijiafeng/home/cuijiafeng/deep_learning/RSKDD-Net/data/poses/10.txt')  # pose
pose = list(f_pose)

kp_dir_path = '/home/cuijiafeng/home/cuijiafeng/deep_learning/RSKDD-Net/test_results/10/keypoints'
kp_txt_list = os.listdir(kp_dir_path)
kp_txt_list.sort()
print(kp_txt_list)
desc_dir_path = '/home/cuijiafeng/home/cuijiafeng/deep_learning/RSKDD-Net/test_results/10/desc'
desc_txt_list = os.listdir(desc_dir_path)
desc_txt_list.sort()

num = 0
for txt_name in kp_txt_list:
    create_json(kp_dir_path + '/' + txt_name, desc_dir_path + '/' + txt_name,
                '/home/cuijiafeng/home/cuijiafeng/deep_learning/SG_PR/data/10/' + txt_name.strip('.txt') + '.json',
                pose[num])
    num += 1
