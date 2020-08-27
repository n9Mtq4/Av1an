from pathlib import Path

from Av1an.chunk import Chunk

TEMP = Path('.temp')


def create_test_chunk():
    chunk = Chunk(TEMP, 1, ['ffmpeg', 'gen', 'cmd'], 'ext', 2, 3)
    chunk.vmaf_target_cq = 32
    return chunk


def test_to_dict():
    chunk = create_test_chunk()
    chunk_dict = chunk.to_dict()

    assert chunk_dict['index'] == chunk.index
    assert chunk_dict['ffmpeg_gen_cmd'] == chunk.ffmpeg_gen_cmd
    assert chunk_dict['size'] == chunk.size
    assert chunk_dict['frames'] == chunk.frames
    assert chunk_dict['output_ext'] == chunk.output_ext
    assert chunk_dict['vmaf_target_cq'] == chunk.vmaf_target_cq


def test_fake_input_path():
    chunk = create_test_chunk()
    assert chunk.fake_input_path == TEMP / 'split' / '00001.mkv'


def test_output_path():
    chunk = create_test_chunk()
    assert chunk.output_path == TEMP / 'encode' / '00001.ext'


def test_output():
    chunk = create_test_chunk()
    assert chunk.output == chunk.output_path.as_posix()


def test_fpf():
    chunk = create_test_chunk()
    assert chunk.fpf == '.temp/split/00001_fpf'


def test_name():
    chunk = create_test_chunk()
    assert chunk.name == '00001'


def test_create_from_dict():
    chunk = create_test_chunk()
    chunk_dict = chunk.to_dict()
    chunk_from_dict = Chunk.create_from_dict(chunk_dict, TEMP)

    assert chunk.__dict__ == chunk_from_dict.__dict__
