from pathlib import Path

from Av1an.arg_parse import Args
from Av1an.chunk import Chunk
from Av1an.commandtypes import CommandPair

TEMP = Path('.temp')


def create_test_chunk():
    chunk = Chunk(TEMP, 1, ['ffmpeg', 'gen', 'cmd'], 'ext', 2, 3)
    chunk.pass_cmds = [
        CommandPair(['ffmpeg', 'filter', 'cmd'], ['encoder', 'cmd', 'pass1']),
        CommandPair(['ffmpeg', 'filter', 'cmd'], ['encoder', 'cmd', 'pass2'])
    ]
    return chunk


def test_generate_pass_cmds():
    chunk = create_test_chunk()
    chunk.pass_cmds = None

    args = Args({})
    args.encoder = 'aom'
    args.passes = 1
    args.ffmpeg_pipe = ['ffmpegpipe1', 'ffmpegpipe1']
    args.video_params = ['video_params1', 'video_params2']

    chunk.generate_pass_cmds(args)

    assert chunk.pass_cmds is not None
    assert isinstance(chunk.pass_cmds, list)
    assert len(chunk.pass_cmds) == 1
    assert isinstance(chunk.pass_cmds[0], CommandPair)
    assert chunk.pass_cmds[0].ffmpeg_cmd[0] == 'ffmpeg'
    assert chunk.pass_cmds[0].encode_cmd[0] == 'aomenc'

    for ffmpeg_pipe_token in args.ffmpeg_pipe:
        assert ffmpeg_pipe_token in chunk.pass_cmds[0].ffmpeg_cmd
    for video_param_token in args.video_params:
        assert video_param_token in chunk.pass_cmds[0].encode_cmd

    args.passes = 2
    chunk.generate_pass_cmds(args)

    assert len(chunk.pass_cmds) == 2


def test_remove_first_pass_from_commands():
    chunk = create_test_chunk()

    assert len(chunk.pass_cmds) == 2

    chunk.remove_first_pass_from_commands()

    assert len(chunk.pass_cmds) == 1
    assert 'pass1' not in chunk.pass_cmds[0].encode_cmd
    assert 'pass2' in chunk.pass_cmds[0].encode_cmd

    pass_before = chunk.pass_cmds[0]
    chunk.remove_first_pass_from_commands()
    assert len(chunk.pass_cmds) == 1
    assert chunk.pass_cmds[0] == pass_before


def test_to_dict():
    chunk = create_test_chunk()
    chunk_dict = chunk.to_dict()

    assert chunk_dict['index'] == chunk.index
    assert chunk_dict['ffmpeg_gen_cmd'] == chunk.ffmpeg_gen_cmd
    assert chunk_dict['size'] == chunk.size
    assert chunk_dict['pass_cmds'] == chunk.pass_cmds
    assert chunk_dict['frames'] == chunk.frames
    assert chunk_dict['output_ext'] == chunk.output_ext


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
