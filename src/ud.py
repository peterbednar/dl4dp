import os
import tarfile
import urllib.request
from io import TextIOWrapper
from utils import progressbar

_TREEBANKS_FILENAME = "ud-treebanks-v2.3.tgz"
_DATASET_FILENAME = "{0}-ud-{1}.conllu"

def _find_filepath(targz, filename):
    for f in targz.getnames():
        if f.endswith("/" + filename):
            return f
    return None

def load_treebank(basename, treebank, dataset="train"):
    if not os.path.isdir(basename):
        os.makedirs(basename)

    tar_path = basename + _TREEBANKS_FILENAME
    if not os.path.isfile(tar_path):
        download_treebanks(tar_path)

    tar = tarfile.open(tar_path, "r:gz")
    f = _find_filepath(tar, _DATASET_FILENAME.format(treebank, dataset))
    if f is None:
        tar.close()
        return None
    else:
        return TextIOWrapper(tar.extractfile(f), encoding="utf-8")

_TREEBANKS_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/" + _TREEBANKS_FILENAME

def download_treebanks(filename, block_size=8192):
    print("downloading {0}".format(_TREEBANKS_FILENAME))

    with urllib.request.urlopen(_TREEBANKS_URL) as r:
        l = int(r.info()["Content-Length"])
        pb = progressbar(l)
        with open(filename, mode="wb") as f:
            while pb.value < l:
                block = r.read(block_size)
                f.write(block)
                pb.update(len(block))
        pb.finish()

import utils

if __name__ == "__main__":
    with load_treebank("../build/", "en_pud", "test") as f:
        for s in utils.read_conllu(f):
            print(s)
