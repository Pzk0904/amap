import json

json_path = './data/amap_traffic_annotations_train.json'

class get_labels():
    def __init__(self):
        self.begin_pos = 0

        with open(json_path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
            data_arr = json_dict["annotations"]
            train_labels = []
            for sample in data_arr:
                train_labels.append(sample['status'])

        self.train_labels = train_labels

    def next_batch_labels(self,batch_size):
        result = self.train_labels[self.begin_pos:self.begin_pos + batch_size]
        self.begin_pos += batch_size
        return result

if __name__ == '__main__':
    aa = get_labels()
    print(type(aa.train_labels[:][1]))

# Traceback (most recent call last):
#   File "./ResNetLSTM1.py", line 148, in <module>
#     a.train()
#   File "./ResNetLSTM1.py", line 112, in train
#     self.tensor1.targets:targets.next_batch_labels(batch_size)
# TypeError: unhashable type: 'list'

