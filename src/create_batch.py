import os
from modules.utils import chunks

dataset_img_path = '../../danbooru/resized/images/'

commands = map(
    lambda x: 'python testing_extract.py % s ../output/%s.json\n' % (x.split('.')[0], x.split('.')[0]),
    os.listdir(dataset_img_path)
)
cmd_chunks = chunks(list(commands), 159)

cnt = 0
for cmd_chunk in cmd_chunks:
    with open('../batch/%d-testing.sh' % cnt, 'w') as fp:
        fp.writelines(cmd_chunk)
    fp.close()

    cnt += 1
