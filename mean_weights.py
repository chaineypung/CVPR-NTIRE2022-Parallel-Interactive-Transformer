import collections
import torch
import os


model_dir = r''
models = []
module_lists = os.listdir(model_dir)
for module in module_lists:
    mode = torch.load(model_dir + module,map_location='cpu')
    models.append(mode)

worker_state_dict = [x['state_dict'] for x in models]
weight_keys = list(worker_state_dict[0].keys())
print(worker_state_dict[0].keys())
fed_state_dict = collections.OrderedDict()
for key in weight_keys:
    key_sum = 0
    for i in range(len(models)):
        key_sum = key_sum + worker_state_dict[i][key]
    fed_state_dict[key] = torch.true_divide(key_sum, len(models))

torch.save({'state_dict': fed_state_dict},'')


