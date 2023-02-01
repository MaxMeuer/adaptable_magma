import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as img
from PIL.Image import DecompressionBombError
from PIL import UnidentifiedImageError
import json
from pathlib import Path
import struct
from itertools import accumulate

from tqdm import tqdm
from typing import List, Tuple, Generator
import random
from multiprocessing import Pool, cpu_count

from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
from torchtyping import TensorType
import traceback
import numpy as np
from PIL import Image
import io
from tokenizers import Tokenizer



dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16,
}

def np_byte_array_to_image(np_byte_array: np.ndarray):
    """
    converts a numpy bytes array to a pillow image
    """
    img_byte_arr = io.BytesIO(np_byte_array.tobytes())
    img = Image.open(img_byte_arr)
    return img

def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def read_jsonl(filename: str) -> Generator[List, None, None]:
    """
    Iterator over data from a jsonl file
    """
    with open(filename) as file:
        for line in file:
            yield json.loads(line.rstrip("\n|\r"))


def read_img_captions(filename: str) -> List[Tuple[str, str]]:
    """
    Yields image_path, image_caption from cc jsonl files
    """
    img_captions = []
    for item in read_jsonl(filename):
        if not "N/A" in item[-2:]:
            img_captions.append((item[-1], item[-2]))
    return img_captions


def load_json(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except Exception:
        print(f"ERROR: Error loading json file {filename}")
        traceback.print_exc()


def _read_image_data(data_dir):
    image_data = []
    img_data_dir = data_dir / "image_data"
    paths = _load_paths(data_dir)
    pbar = tqdm(
        paths,
        desc=f"loading dataset from {str(data_dir)}",
    )
    # read data with multiprocessing
    with Pool(cpu_count()) as pool:
        for img_data in pool.imap(load_json, pbar):
            if img_data is not None:
                image_data.append(img_data)
    return image_data


def _load_paths(data_dir, sort=True):
    paths = []
    img_data_dir = data_dir / "image_data"
    for p in tqdm(
        Path(img_data_dir).glob("*/*.json"),
        desc=f"loading dataset paths from {str(data_dir)}",
    ):
        paths.append(p)
    return sorted(paths)

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                # print("    warming up index mmap file...", flush=True)
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            # print("    reading sizes...", flush=True)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            # print("    reading pointers...", flush=True)
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            # print("    reading document index...", flush=True)
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        # @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state, True)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            # print("    warming up data mmap file...", flush=True)
            _warmup_mmap_file(data_file_path(self._path))
        # print("    creating numpy buffer of mmap...", flush=True)
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        # print("    creating memory view of numpy buffer...", flush=True)
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        try:
            self._bin_buffer_mmap._mmap.close()
        except:
            raise ValueError(self._path)
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
            )
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )

class LazyLoader:
    def __init__(self, data_dir, few_shot):
        self.few_shot = few_shot
        self.paths = _load_paths(data_dir)

    def __len__(self):
        return len(self.paths)//self.few_shot

    def __getitem__(self, idx):
        all_data = []
        for i in range(self.few_shot):
            data = load_json(self.paths[self.few_shot*idx + i])
            all_data.append(data)
        if data is None:
            return self[random.randint(0, len(self) - 1)]
        return all_data


class ImgCptDataset(Dataset):
    """
    Dataset which loads image caption data from our standard format and transforms them into tensors that can be input to the model.
    Images are expected to be stored in data_dir/images, image data in data_dir/image_data and each data item is a json file with format {"image_path": img_path, "captions": [caption1, caption2,...], "metadata":{...}}
    """

    def __init__(
        self, data_dir, tokenizer, transforms, seq_len=2048, load_data_in_memory=False, few_shot=1
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.seq_len = seq_len
        self.load_data_in_memory = load_data_in_memory
        self.few_shot = few_shot

        self.orig_tokenizer = Tokenizer.from_file('/root/bjoern/transformer/tests/files/alpha-001-128k.json')
        self.image_data = MMapIndexedDataset(str(self.data_dir)+'_images')
        self.text_data = MMapIndexedDataset(str(self.data_dir)+'_tokenizer_alpha-001-128k_eng')

    def __len__(self):
        return len(self.text_data)//self.few_shot

    def __getitem__(
        self, idx
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"]]:
        cur_data = self.text_data[self.few_shot *idx: (self.few_shot * idx) + self.few_shot]

        try:

            images = []
            textes = []
            text = ""
            for i, t in enumerate(cur_data):
                try:
                    cur_img = np_byte_array_to_image(self.image_data[int(t[0])])
                except:
                    continue
                # tokenized t[0] is index of image, t[1:] is actual caption

                text = f'<|image|> {self.orig_tokenizer.decode(t[1:])} '

                images.append(self.transforms(cur_img))
                textes.append(self.tokenizer.encode(
                    text,
                    return_tensors="pt",
                    max_length=self.seq_len,
                    padding="max_length",
                    truncation=True,
                ))

            img_tensor = torch.cat(images)
            caption_tensor = torch.cat(textes)

            return img_tensor, caption_tensor
        except (
            UnidentifiedImageError,
            OSError,
            DecompressionBombError,
            IndexError,
        ) as e:
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(img_path)}")
            return self[random.randint(0, len(self) - 1)]


def collate_fn(batch_data: List[Tuple[torch.Tensor, torch.Tensor]], seq_len=2048):

    all_images, all_captions = list(
        zip(*batch_data)
    )  # [(img1, caption1), (img2, caption2), ... ] -> [(img1, img2, ... ), (caption1, caption2, ... )]
    return torch.cat(all_images), torch.cat([i[:, :seq_len] for i in all_captions])
