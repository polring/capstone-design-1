import struct

from src.exporter import MAGIC, export_state_dict


def test_export_round_trip_header_and_tensor(tmp_path):
    output = tmp_path / "dummy.bin"
    export_state_dict({"layer.weight": [[1.0, 2.0], [3.0, 4.0]]}, output)

    data = output.read_bytes()
    assert data[:4] == MAGIC
    version, count = struct.unpack_from("<II", data, 4)
    assert (version, count) == (1, 1)
    name_length = struct.unpack_from("<I", data, 12)[0]
    cursor = 16 + name_length
    ndim = struct.unpack_from("<I", data, cursor)[0]
    cursor += 4
    assert ndim == 2
    assert struct.unpack_from("<II", data, cursor) == (2, 2)
