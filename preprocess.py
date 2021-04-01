# coding:UTF-8
import os
import json
import string

'''
created by cyf 2021-01-19
create a json file of points pair from raw data
'''

def create_json(input_txt_path, output_josn_path, pose,num):
    data = json.load(open('/home/cuijiafeng/home/cuijiafeng/deep_learning/SG_PR/data/0.json'))
    f = open(input_txt_path)  # 特征点位置
    s_dof = list(f) # s_dof返回一个1024个元素的列表，每个元素是一行数据的字符串
    data['centers'].clear()
    data['nodes'].clear()
    data['pose'].clear()

    for i in range(len(s_dof)):
        s = s_dof[i].split(' ', -1) # 将一行字符串按空格分割
        s = s[0:3]
        s[0] = float(s[0])
        s[1] = float(s[1])
        s[2] = float(s[2])

        data['centers'].append(s)
        data['nodes'].append(0)

        # print(s)
        # break

    pose_str = pose.split(' ', -1)
    for i in range(len(pose_str)):
        data['pose'].append(float(pose_str[i]))

    with open(output_josn_path, 'w') as file_obj:
        json.dump(data, file_obj)


f_pose = open('/home/cuijiafeng/home/cuijiafeng/deep_learning/RSKDD-Net/data/poses/04.txt')
pose = list(f_pose)
dir_path = '/home/cuijiafeng/home/cuijiafeng/deep_learning/RSKDD-Net/test_results/04/keypoints'
txt_list = os.listdir(dir_path) # 返回当前目录下的文件或文件夹
txt_list.sort()

num = 0
# .strip() 去除字符串首尾的括号中指定的字符串内容
for txt_name in txt_list:
    create_json(dir_path + '/' + txt_name,
                '/home/cuijiafeng/home/cuijiafeng/deep_learning/SG_PR/data/04/' + txt_name.strip('.txt') + '.json',
                pose[num],num)
    num += 1
