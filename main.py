import fire
from manga_cleaner import MangaCleaner

class Runner():
    def start(self, file):
        file = str(file) + '.jpg'
        print('Input file: ' + file)
        mangaCleaner = MangaCleaner(file)

if __name__ == '__main__':
    fire.Fire(Runner)
