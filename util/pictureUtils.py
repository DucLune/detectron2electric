# %%
import os
import random
import shutil


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
def buildCocoDataset(pics_dir, anno_dir, dest_dir, train_portion=0.7, val_portion=0.15, test_portion=0.15):
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "test"), exist_ok=True)
    list_data = os.listdir(pics_dir)
    random.shuffle(list_data)
    list_train = list_data[:int(train_portion*len(list_data))]
    list_val = list_data[int(train_portion*len(list_data))                         :int((train_portion+val_portion)*len(list_data))]
    list_test = list_data[int((train_portion+val_portion)*len(list_data)):]
    count = 0
    for filename in list_train:
        shutil.copy(os.path.join(pics_dir, filename),
                    os.path.join(dest_dir, "train"))
        if os.path.exists(os.path.join(anno_dir, filename.split('.')[0]+".json")):
            shutil.copy(os.path.join(anno_dir, filename.split('.')[
                        0]+".json"), os.path.join(dest_dir, "train"))

    for filename in list_val:
        shutil.copy(os.path.join(pics_dir, filename),
                    os.path.join(dest_dir, "val"))
        if os.path.exists(os.path.join(anno_dir, filename.split('.')[0]+".json")):
            shutil.copy(os.path.join(anno_dir, filename.split('.')[
                        0]+".json"), os.path.join(dest_dir, "val"))
    for filename in list_test:
        shutil.copy(os.path.join(pics_dir, filename),
                    os.path.join(dest_dir, "test"))
        if os.path.exists(os.path.join(anno_dir, filename.split('.')[0]+".json")):
            shutil.copy(os.path.join(anno_dir, filename.split('.')[
                        0]+".json"), os.path.join(dest_dir, "test"))
                        
#buildCocoDataset("../t", "../t", "../mycoco")



