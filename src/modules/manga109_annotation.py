import xml.etree.ElementTree as Et


class Manga109Annotation:
    def __init__(self, path_xml: str, page_index: int = 0):
        self.root = Et.parse(path_xml).getroot()
        self.page_index = page_index

    def select_page(self, page_number: int) -> None:
        self.page_index = page_number

    def get_text_area_list(self) -> list:
        page = self.root.find('pages')[self.page_index]
        pts = []

        for text in page.findall('text'):
            topleft_pt = (int(text.get('ymin')), int(text.get('xmin')))
            bottomright_pt = (int(text.get('ymax')), int(text.get('xmax')))
            pts.append((topleft_pt, bottomright_pt))

        return pts

    def get_text_area_dict(self) -> list:
        page = self.root.find('pages')[self.page_index]
        pts = {}

        for text in page.findall('text'):
            pts[text.get('id')] = {
                'topleft_pt': (int(text.get('ymin')), int(text.get('xmin'))),
                'bottomright_pt': (int(text.get('ymax')), int(text.get('xmax')))
            }

        return pts
