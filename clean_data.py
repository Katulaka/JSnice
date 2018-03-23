import json

f_train = 'mljs/jsnice_data/training_processed.txt'
f_test =  'mljs/jsnice_data/eval_processed.txt'
f_in = [f_train, f_test]

max_context = 100
out_train =  'train_data_{}.json'.format(max_context)
out_test =  'test_data_{}.json'.format(max_context)
f_out = [out_train, out_test]

for path_r, path_w in zip(f_in, f_out):
    with open(path_r) as js_file:
        data = js_file.readlines()
        clean_data = []
        for d in data:
            ds = json.loads(d)
            if len(ds['input']) <= max_context:
                inp = [[el for el in dd if el!= '0PAD'] for dd in ds['input']]
                clean_data.append((inp, ds['output']))

    with open(path_w, 'w') as outfile:
        json.dump(clean_data, outfile)
