import os
from modules.utils import chunks

dataset_img_path = '../../danbooru/resized/images/'

commands = map(
    lambda x: 'python save_label.py % s ../output/%s.json\n' % (x.split('.')[0], x.split('.')[0]),
    os.listdir(dataset_img_path)
)
cmd_chunks = chunks(list(commands), 8)

cnt = 0
for cmd_chunk in cmd_chunks:
    with open('../batch/%d.sh' % cnt, 'w') as fp:
        fp.write("'#!/bin/bash'\n")
        fp.writelines(cmd_chunk)
    fp.close()

    cnt += 1
