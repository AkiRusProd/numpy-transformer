import tarfile
import requests
from tqdm import tqdm

#References: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/datasets/multi30k.html

urls=[
     'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
     'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
     'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz'
 ]

path = "dataset/"
filenames = ["mmt16_task1_test.tar.gz", "training.tar.gz", "validation.tar.gz"]


def download(urls, path, filenames):
    for _, (url, filename) in enumerate(zip(urls, filenames)):
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))
        with open(path + filename, 'wb') as file, tqdm(
                desc = f'downloading {filename = } to {path = }',
                total = total,
                unit = 'iB',
                unit_scale = True,
                unit_divisor = 1024,
        ) as bar:
            for data in resp.iter_content(chunk_size = 1024):
                size = file.write(data)
                bar.update(size)

download(urls, path, filenames)


def extract(path, filenames):
    for filename in filenames:
        tar = tarfile.open(path + filename)
        tar.extractall(path)
        tar.close()

        print(f'Extracted {filename}')


extract(path, filenames)