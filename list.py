import os
"""
This script creates a txt file for ycb video data set to show the objects in every video
"""
def main():
    d = open("list.txt", "w")
    # take the first txt file in every folder
    for i in range(0, 92):
        if i < 10:
            txt_path = os.path.join("../YCB_Video_Dataset/data/000{}".format(i), "000001-box.txt")
            if os.path.exists(txt_path) is False:
                print("warning:file does't exist, number 000{}".format(i))
            f = open(txt_path, "r")

        else:
            txt_path = os.path.join("../YCB_Video_Dataset/data/00{}".format(i), "000001-box.txt")
            if os.path.exists(txt_path) is False:
                print("warning:file does't exist, number 00{}".format(i))
            f = open(txt_path, "r")
        # read the txt file and write the objects in every file into a line
        lines = f.readlines()
        d.write("{}".format(i))
        for index, line in enumerate(lines):
            if index == len(lines) - 1:
                d.write(", {}".format(line.split("_")[0]) + '\n')
            else:
                d.write(", {}".format(line.split("_")[0]))

if __name__ == "__main__":
    main()
    print("list created, open with txt or excel")
