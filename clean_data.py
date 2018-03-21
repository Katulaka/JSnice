import json
path = 'mljs/jsnice_data/training_processed.txt'
with open(path) as train_file:
    train = train_file.readlines()
train_data = []
max_context = 20
for t in train:
    ts = json.loads(t)
    if len(ts['input']) <= max_context:
        inp = [[el for el in tt if el!= '0PAD'] for tt in ts['input']]
        train_data.append((inp, ts['output']))
with open('train_data.json', 'w') as outfile:
    json.dump(train_data, outfile)
