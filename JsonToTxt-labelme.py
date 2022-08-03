import json
import os
import numpy as np
import cv2

points_num = 6  # labelme多边形标注时是几边形
dictionary = dict([('bb', 0), ('rb', 1)])  # 如果labelme标定时类别名非数字编号，需要用dictionary映射；反之该行无用
source = 'W:\\net\\RM-Neural-Network-2022\\NCNET\\enginner\\blue4-sixpoints\\'  # 源目录，该目录下应有图片和对应的json文件
target = 'W:\\net\\RM-Neural-Network-2022\\NCNET\\enginner\\exchange-3-6-ori\\'  # 目标目录，该目录下会生成图片和对应的txt文件，每行格式为 class x1 y1 x2 y2 ....xn yn 共2*n+1个数字

names = os.listdir(source)
for name in names:
    if len(name.split('.')) > 1:
        if name.split('.')[1] == 'json':
            items = []
            obj = json.load(open(source + name))
            shape = obj['shapes']
            h = obj['imageHeight']
            w = obj['imageWidth']
            imgdir = obj['imagePath']
            img = cv2.imread(source + imgdir)

            for item in shape:
                label = int(item['label'])  # 如果labelme标定时类别名非数字编号，需要用dict映射

                points = np.array(item['points']).flatten() / [*[w, h] * points_num]
                line = np.array([label, *points])
                #print(line.shape)
                items.append(line)
            items = np.array(items).astype(np.float16)
            #print(items)
            cv2.imwrite(target + 'images\\' + imgdir, img)
            np.savetxt(target + 'labels\\' + name.replace(".json", ".txt"), items, fmt='%f')
