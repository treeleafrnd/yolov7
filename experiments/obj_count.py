import os
def count_objects(path):
    label = {}
    for files in os.listdir('path_to_label'):
        key = os.path.splitext(files)[0]
        # print(key)
        with open('path_to_label' + files, 'r') as fp:
            line_count = len(fp.readlines())
        #    print(line_count)
        label[key] = line_count

    label_pred = {}
    for files in os.listdir(path):
        key = os.path.splitext(files)[0]
        # print(key)
        with open(path+'/'+ files,'r') as fp:
            line_count = len(fp.readlines())
        #    print(line_count)
        label_pred[key] = line_count

    print('number of labeled images:',len(label),
          '\nnumber of pred images: ',len(label_pred),
          '\nnumber of images with zero objects: ',len(label)-len(label_pred))
    print(sum(label.values()))
    print(sum(label_pred.values()))

#count_object('path to predicted label file')

count_objects('path')