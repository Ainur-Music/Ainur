from audio_data_pytorch import MetaDataset, WAVDataset, AllTransform
from audio_data_pytorch.datasets.wav_dataset import get_all_wav_filenames
from mutagen import File
import regex as re
from datetime import datetime
import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def select(key, **kwargs):
    value = kwargs.get(key, None)
    return value

def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


def prefix_dict(prefix: str, d: Dict) -> Dict:
    return {prefix + str(k): v for k, v in d.items()}

def isclose_datetime(t1, t2, rel_tol=0, abs_tol=1.0):

    # combine each time object with a date to create a datetime object
    dt1 = datetime.combine(datetime.today(), t1)
    dt2 = datetime.combine(datetime.today(), t2)
    # Calculate the difference between the two datetimes
    diff = abs((dt1 - dt2).total_seconds())
    
    # Calculate the maximum allowed difference based on the relative and absolute tolerances
    max_diff = max(rel_tol * diff, abs_tol)
    
    # Check if the difference is within the maximum allowed difference
    return diff <= max_diff

def is_in_time_range(time, start_t, end_t, abs_tol=1.0):
    t = time[0].strip("[]")
    if int(t.split(":")[0]) < 60:
        return (start_t <= datetime.strptime(t, "%M:%S.%f").time() <= end_t or
                 isclose_datetime(datetime.strptime(t, "%M:%S.%f").time(), start_t, abs_tol=abs_tol))
    else:
        return False


class LyricsDataset(MetaDataset):
    def __init__(self, path, **kwargs):
        wav_kwargs, kwargs  = groupby("wav_", kwargs)
        super().__init__(path=path, metadata_mapping_path=None, **wav_kwargs)
        self.sample_rate = select("sample_rate", **wav_kwargs)
        self.crop_size = select('crop', **kwargs)
        self.time_range = self.crop_size / self.sample_rate
        self.wavs = get_all_wav_filenames(path, recursive=False)

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, idx):
        audio, artist, genre = super().__getitem__(idx)
        lyrics = File(self.wavs[idx]).get('lyrics', [""])[0]
        pattern = r"(?<=\[\d\d\:\d\d\.\d\d\d\])|(?<=\[\d\d\:\d\d\.\d\d\])"
        start = np.random.randint(0, audio.shape[-1] - self.crop_size)
        end = start + self.crop_size
        if audio.shape[-1] < self.crop_size:
            start = 0
            end = audio.shape[-1]
        start_t = datetime.utcfromtimestamp(start / self.sample_rate).time()
        end_t = datetime.utcfromtimestamp(end / self.sample_rate).time()
        time_lyrics = "\n".join(
            map(lambda t: t[1],    
                filter(lambda t : len(t) == 2 and is_in_time_range(t, start_t, end_t), 
                        map(lambda line : tuple(re.split(pattern, line, maxsplit=1)), lyrics.split("\n"))
                    )
                )
            )
        return audio[:,start:end], artist, genre, time_lyrics
        

        

def get_dataset(path, sample_rate=48000, crop=2**20, wav_loudness=None, wav_scale=None, wav_stereo=True, wav_mono=False):
    wav_transforms = AllTransform(
        source_rate = None,
        target_rate = None,
        crop_size = None,
        random_crop_size = None,
        loudness = wav_loudness,
        scale = wav_scale,
        mono = wav_mono,
        stereo = wav_stereo,
    )
    


    return LyricsDataset(
        path = [path], # Path or list of paths from which to load files
        crop = crop,
        #**kwargs Forwarded to `MetaDataset`
        wav_recursive = False, # Recursively load files from provided paths
        wav_sample_rate = sample_rate, # Specify sample rate to convert files to on read
        wav_transforms = wav_transforms, # Transforms to apply to audio files
        wav_check_silence = True # Discards silent samples if true
    )



if __name__ == "__main__":
    dataset = get_dataset("/Users/gio/spotdl/", crop=2**20, wav_mono=True, wav_stereo=False)
    for audio, artist, genre, lyrics in dataset:
        print(lyrics)
        #sd.play(audio[0,:].numpy(), 48000)
        #sd.wait()  # Wait until the audio has finished playing