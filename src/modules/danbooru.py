import json


class Danbooru:
    def __init__(self, id):
        with open('../../danbooru/resized/annotations/%s.json' % id, 'r') as fp:
            self.raw = fp.read()

        fp.close()

    def get_text_area(self) -> list:
        return json.loads(self.raw)
