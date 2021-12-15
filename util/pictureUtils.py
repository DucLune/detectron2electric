# %%
import os
import random
import shutil
import labelme2coco
import glob

# %%
# src_dirname:图片存储目录
# componentType零件类型
# componentCount零件数量
# startIndex起始零件编号
# startPosition起始面编号
# positions面的数量
# prefix文件名前缀
def namepictures(src_dirname, componentType, componentCount, startIndex=1, startPosition=1, positions=4, prefix="Image"):
    list_data = os.listdir(src_dirname)
    list_data.sort()
    currentIndex = startIndex
    currentPosition = startPosition
    count = 0
    for filename in list_data:
        if filename[:5] == prefix:
            title = filename.split(".")
            #print(componenttype+str(currentIndex) +'-'+str(currentPosition)+'.'+title[1])
            print("Renaming "+os.path.join(src_dirname, filename)+" -> "+os.path.join(src_dirname, componentType+str(currentIndex) + '-'+str(currentPosition)+'.'+title[1]))
            os.rename(os.path.join(src_dirname, filename), os.path.join(
                src_dirname, componentType+str(currentIndex) + '-'+str(currentPosition)+'.'+title[1]))
            currentPosition += 1
            if currentPosition == startPosition+positions:
                currentPosition = startPosition
                currentIndex += 1
                count += 1
                if count == componentCount:
                    break
    return


# %%
#namepictures("../t","D",8)

# %%
import cv2


def bmp2jpg(src_dirname):
    list_data = os.listdir(src_dirname)
    for filename in list_data:
        if filename.split('.')[1] != 'bmp':
            continue
        im = cv2.imread(os.path.join(src_dirname, filename))
        print("Converting "+os.path.join(src_dirname, filename)+" to " +
              os.path.join(src_dirname, filename.split('.')[0]+'.jpg'))
        cv2.imwrite(os.path.join(
            src_dirname, filename.split('.')[0]+'.jpg'), im)


#bmp2jpg("../t")


# %%
# 图片和标注放在两个不同的文件夹
# 图片放在pics_dir中，标注放在anno_dir中

def buildCocoDataset(anno_src_dir,dataset_dir,labels=[], train_portion=0.7, val_portion=0.15, test_portion=0.15):
    os.makedirs(dataset_dir, exist_ok=True)
    anno_dest_dir = os.path.join(dataset_dir, "annotations")
    os.makedirs(anno_dest_dir, exist_ok=True)
    train_dir = os.path.join(dataset_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(dataset_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(dataset_dir, "test")
    os.makedirs(test_dir, exist_ok=True)
    list_data = os.listdir(anno_src_dir)
    random.shuffle(list_data)
    list_train = list_data[:int(train_portion*len(list_data))]
    list_val = list_data[int(train_portion*len(list_data)):int((train_portion+val_portion)*len(list_data))]
    list_test = list_data[int((train_portion+val_portion)*len(list_data)):]
    count = 0
    for filename in list_train:
        shutil.copy(os.path.join(anno_src_dir, filename),train_dir)
    for filename in list_val:
        shutil.copy(os.path.join(anno_src_dir, filename),val_dir)
    for filename in list_test:
        shutil.copy(os.path.join(anno_src_dir, filename),test_dir)
    labelme2coco.labelme2coco(glob.glob(os.path.join(train_dir,"*.json")),os.path.join(anno_dest_dir,"train.json"),labels)
    labelme2coco.labelme2coco(glob.glob(os.path.join(val_dir,"*.json")),os.path.join(anno_dest_dir,"val.json"),labels)
    labelme2coco.labelme2coco(glob.glob(os.path.join(test_dir,"*.json")),os.path.join(anno_dest_dir,"test.json"),labels) 
#
#print(os.getcwd())
#buildCocoDataset("../Annotations/glue_pin_inclined_side_pin_glue_injection_hole_regular_part","../dataset/coco-20211215",["glue","injection_hole"])



