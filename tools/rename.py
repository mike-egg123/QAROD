import os

def get_filepaths(parent):
    filepaths = []
    for filepath, dirnames, filenames in os.walk(parent):
        for filename in filenames:
            filepaths.append(os.path.join(filepath, filename))
    return filepaths

if __name__ == '__main__':
    for filepath, dirnames, filenames in os.walk(r"E:\文档\课程\大四\毕设\程序\data\banpei"):
        for filename in filenames:
            new_filename = "banpei_" + filename
            os.rename(os.path.join(filepath, filename), os.path.join(filepath, new_filename))
