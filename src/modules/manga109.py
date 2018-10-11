import xml.etree.ElementTree as Et


class Manga109:
    def __init__(self, title: str):
        annotation_path = '../../Manga109-small/annotations/' + title + '.xml'
        self.root = Et.parse(annotation_path).getroot()

    def get_text_area(self, page_id: int) -> list:
        page = self.root.find('pages')[page_id]
        data = []

        for text in page.findall('text'):
            y1, x1 = int(text.get('ymin')), int(text.get('xmin'))
            y2, x2 = int(text.get('ymax')), int(text.get('xmax'))
            width, height = x2 - x1, y2 - y1

            yield {'x': x1, 'y': y1, 'width': width, 'height': height}
