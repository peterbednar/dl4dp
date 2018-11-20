import urllib.request

from utils import progressbar

_TREEBANKS_FILENAME = "ud-treebanks-v2.3.tgz"
_TREEBANKS_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2895/" + _TREEBANKS_FILENAME

_BLOCK_SIZE = 8192

def download_treebanks(basename):
    with urllib.request.urlopen(_TREEBANKS_URL) as r:
        l = int(r.info()["Content-Length"])
        pb = progressbar(l)
        with open(basename + _TREEBANKS_FILENAME, mode="wb") as f:
            while pb.value < l:
                block = r.read(_BLOCK_SIZE)
                f.write(block)
                pb.update(len(block))
        pb.finish()

if __name__ == "__main__":
    download_treebanks("../build/")
