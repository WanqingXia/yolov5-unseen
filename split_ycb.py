"""
This script has two functions:
1. Combine original train and val folder and split them into seen/unseen folder
2. Create all the files needed for training (.names data.data .shape .txt hyp cfg)
"""
import os
import random
import csv

from tqdm import tqdm
import json
import shutil

# root directory of YCB dataset
dataset_root = "../YCB_Video_Dataset"
data_root = os.path.join(dataset_root, "data")
names_root = os.path.join(dataset_root, "image_sets", "trainval.txt")
class_root = os.path.join(dataset_root, "image_sets", "classes.txt")

# Check whether all the original directories exist
assert os.path.exists(data_root), "YCB data path not exist..."
assert os.path.exists(names_root), "YCB names path not exist..."
assert os.path.exists(class_root), "YCB class path not exist..."

# save file directory (all train and val image, depth, labels )
save_file_root = "../ycb_unseen"
train_path = os.path.join(save_file_root, "train")
val_path = os.path.join(save_file_root, "val")
test_path = os.path.join(save_file_root, "test")

# Concatenate the path for saving all the data (images, depth, labels)
train_image_path = os.path.join(train_path, "images")
train_label_path = os.path.join(train_path, "labels")
train_depth_path = os.path.join(train_path, "depth")
val_image_path = os.path.join(val_path, "images")
val_label_path = os.path.join(val_path, "labels")
val_depth_path = os.path.join(val_path, "depth")
test_image_path = os.path.join(test_path, "images")
test_label_path = os.path.join(test_path, "labels")
test_depth_path = os.path.join(test_path, "depth")

# make all the directories
if os.path.exists(save_file_root) is False:
    # parent level
    os.makedirs(save_file_root)
    os.makedirs(train_path)
    os.makedirs(val_path)
    os.makedirs(test_path)

    # child level
    os.makedirs(train_image_path)
    os.makedirs(train_label_path)
    os.makedirs(train_depth_path)
    os.makedirs(val_image_path)
    os.makedirs(val_label_path)
    os.makedirs(val_depth_path)
    os.makedirs(test_image_path)
    os.makedirs(test_label_path)
    os.makedirs(test_depth_path)
"""
6 mustard bottle [4]
9 gelatin box [7]
19 pitcher base [10]
35 power drill [14]
any file contains those four category will be allocated in to val
"""

def choose_files(classes_file):
    train_val_file = open(names_root, "r").readlines()
    # create train_data.txt val_data.txt to store all the names
    train_val_split = open(os.path.join(save_file_root, "train_val_split.txt"), 'w')
    test_split = open(os.path.join(save_file_root, "test_split.txt"), 'w')
    class_dict = open(classes_file, 'r').readlines()
    num_test = 0
    num_train = 0
    num_val = 0
    # For every file in the dictionary, put images that only have seen objects into train,
    # images that only have unseen objects into val
    for i, file in tqdm(enumerate(train_val_file), desc='Spliting files', total=len(train_val_file)):
        with open(os.path.join(data_root, file.strip('\n') + "-box.txt"), "r") as l:
            label = l.readlines()
            all_to_val = False
            for index, line in enumerate(label):
                if class_dict[4].strip('\n') in line or class_dict[7].strip('\n') in line \
                        or class_dict[10].strip('\n') in line or class_dict[14].strip('\n') in line:
                    all_to_val = True

            if all_to_val:
                test_split.write(file)
                num_test += 1
            else:
                train_val_split.write(file)

    train_val_split.close()
    test_split.close()

    train_split = open(os.path.join(save_file_root, "train_split.txt"), 'w')
    val_split = open(os.path.join(save_file_root, "val_split.txt"), 'w')
    random.seed(0)
    with open(os.path.join(save_file_root, "train_val_split.txt"), 'r') as r:
        train_val_file = r.readlines()

    for i, file in tqdm(enumerate(train_val_file), desc='Spliting train_val', total=len(train_val_file)):
        num = random.randrange(0.0, 5.0)
        if num < 1.0:
            val_split.write(file)
            num_val += 1
        else:
            train_split.write(file)
            num_train += 1

    print("\nnumber of train file chosen: {}".format(num_train))
    print("\nnumber of val file chosen: {}".format(num_val))
    print("\nnumber of test file chosen: {}".format(num_test))
    print("\nfile choosing finished")

"""
Copy files from the original folder to our division folder train/val
Record the new path for every image in a txt file
"""
def copy_files(task):
    if task == "train":
        dict_ini = open(os.path.join(save_file_root, "train_split.txt"), 'r').readlines()
        task_image_path = train_image_path
        task_depth_path = train_depth_path
        task_data_record = open(os.path.join(save_file_root, "train_data.txt"), 'w')
    elif task == "val":
        dict_ini = open(os.path.join(save_file_root, "val_split.txt"), 'r').readlines()
        task_image_path = val_image_path
        task_depth_path = val_depth_path
        task_data_record = open(os.path.join(save_file_root, "val_data.txt"), 'w')
    elif task == "test":
        dict_ini = open(os.path.join(save_file_root, "test_split.txt"), 'r').readlines()
        task_image_path = test_image_path
        task_depth_path = test_depth_path
        task_data_record = open(os.path.join(save_file_root, "test_data.txt"), 'w')

    # loop through all file names in the train/val_split.txt
    for i, name in tqdm(enumerate(dict_ini), desc='Copying {} files over...'.format(task), total=len(dict_ini)):
        file = name.strip('\n')
        file_path = os.path.join(data_root, file)
        # find color and depth images in their original path
        img_file = file_path + '-color.png'
        dep_file = file_path + '-depth.png'
        assert os.path.exists(img_file), "\n warning: image file {} not exist...".format(img_file)
        assert os.path.exists(dep_file), "\n warning: depth file {} not exist...".format(dep_file)
        # create new path and copy image files over
        img_path_copy_to = os.path.join(task_image_path, file.split('/')[0] + "-" + file.split('/')[1] + "-color.png")
        task_data_record.write(img_path_copy_to + "\n")
        if os.path.exists(img_path_copy_to) is False:
            shutil.copyfile(img_file, img_path_copy_to)
        dep_path_copy_to = os.path.join(task_depth_path, file.split('/')[0] + "-" + file.split('/')[1] + "-depth.png")
        if os.path.exists(dep_path_copy_to) is False:
            shutil.copyfile(dep_file, dep_path_copy_to)
    print('\nCopy {} files finished'.format(task))
    task_data_record.close()

"""
translate from ycb style labelling to yolo style
1. change object name to its index number in my_class.txt
2. change box boundary from absolute value [xmin, ymin, xmax, ymax] to [x, y, w, h]
"""
def translate_info(classes_file, task, label_count):
    # All images in ycb are the same size
    img_height = 480
    img_width = 640
    # read class file
    class_dict = open(classes_file, 'r').readlines()
    count = [0]*21
    # write to different files for train dataset and val dataset
    with open("attributes.csv","r") as file:
        csv_file = csv.reader(file)
        header = next(csv_file)
        attribute_matrix = []
        for row in csv_file:
           attribute_matrix.append(row[1:])

    if task == "train":
        dict_ini = open(os.path.join(save_file_root, "train_split.txt"), 'r').readlines()
        txt_save_path = train_label_path
        txt_dict = open(os.path.join(save_file_root, "train_labels.txt"), 'w')
    elif task == "val":
        dict_ini = open(os.path.join(save_file_root, "val_split.txt"), 'r').readlines()
        txt_save_path = val_label_path
        txt_dict = open(os.path.join(save_file_root, "val_labels.txt"), 'w')
    elif task == "test":
        dict_ini = open(os.path.join(save_file_root, "test_split.txt"), 'r').readlines()
        txt_save_path = test_label_path
        txt_dict = open(os.path.join(save_file_root, "test_labels.txt"), 'w')
    # record how many files have bad data
    bad_data_count = 0
    # loop through all file names in train/val_split.txt
    for i, name in tqdm(enumerate(dict_ini), desc='Translating {} labels...'.format(task), total=len(dict_ini)):
        file = name.strip('\n')
        file_path = os.path.join(data_root, file)
        txt_file = file_path + '-box.txt'
        assert os.path.exists(txt_file), "\n warning: txt file {} not exist...".format(txt_file)
        # rename the txt file based on its original path and name
        txt_save_name = os.path.join(txt_save_path, file.split('/')[0] + "-" + file.split('/')[1] + "-box.txt")
        # read the information for every object in the label file
        with open(txt_save_name, "w") as f, open(txt_file, "r") as r:
            for index, obj in enumerate(r.readlines()):
                # get the bounding box for each object
                xmin = float(obj.split(" ")[1])
                ymin = float(obj.split(" ")[2])
                xmax = float(obj.split(" ")[3])
                ymax = float(obj.split(" ")[4])
                # find the corresponded index for the object name
                class_name = obj.split(" ")[0].strip()[4:]
                index_num = 0
                for class_index, classes in enumerate(class_dict):
                    if class_name == classes.strip('\n'):
                        index_num = class_index
                        count[index_num] += 1
                        break
                # check the data to make sure max is always bigger than min, which will affect the calculation of loss
                if xmax <= xmin or ymax <= ymin:
                    bad_data_count += 1
                    continue
                # translate bounding box to YOLO format (centre x, centre y, width, height)
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # real coordinate to relative coordinate (relative to the size of image), 6 decimal places
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)
                info = [str(num) for num in [index_num, xcenter, ycenter, w, h]]

                attr= [str(num) for num in attribute_matrix[index_num]]
                joined_str = " ".join(attr)

                f.write(" ".join(info))
                f.write(" " + joined_str + '\n')
        # Check if there are empty txt file created
        if os.stat(txt_save_name).st_size is 0:
            print("\n file {} is empty".format(txt_save_name))
        txt_dict.write(txt_save_name + "\n")

    labels = [str(num) for num in count]
    label_count.write(" ".join(labels) + '\n')
    print("\n{} bad data point found and removed".format(bad_data_count))
    print("\nConverting {} label finished!".format(task))


def create_data_data(classes_file):
    data_data = open(os.path.join(save_file_root, "ycb_data.data"), 'w')
    num_classes = len(open(classes_file, 'r').readlines())
    data_data.write("classes={}\n".format(num_classes))
    data_data.write("train=" + save_file_root + "/train_data.txt\n")
    data_data.write("valid=" + save_file_root + "/val_data.txt\n")
    data_data.write("test=" + save_file_root + "/test_data.txt\n")
    data_data.write("names=" + save_file_root + "/my_classes.names\n")

def main():
    # read classes from classes.txt and translate to my_classes
    classes = open(class_root, 'r').readlines()
    new_class_dict = os.path.join(save_file_root, "my_classes.names")
    new_class = open(new_class_dict, 'w')
    for index, line in enumerate(classes):
        if index + 1 == len(classes):
            new_class.write("{}".format(line.strip()[4:]))
        else:
            new_class.write("{}".format(line.strip()[4:]) + "\n")
    new_class.close()

    choose_files(new_class_dict)

    label_count = open(os.path.join(save_file_root, "label_count.txt"), "w")
    # Translate YCB labelling to yolo labelling
    translate_info(new_class_dict, "train", label_count)
    translate_info(new_class_dict, "val", label_count)
    translate_info(new_class_dict, "test", label_count)
    label_count.close()

    # Copy image and depth file to the new directory
    copy_files("train")
    copy_files("val")
    copy_files("test")

    # create data.data file
    create_data_data(new_class_dict)

if __name__ == '__main__':
    main()
