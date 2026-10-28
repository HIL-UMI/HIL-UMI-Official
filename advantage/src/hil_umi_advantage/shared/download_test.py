import pathlib

import pytest

import hil_umi_advantage.shared.download as download


@pytest.fixture(scope="session", autouse=True)
def set_openpi_data_home(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("openpi_data")
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENPI_DATA_HOME", str(temp_dir))
        yield


def test_download_local(tmp_path: pathlib.Path):
    local_path = tmp_path / "local"
    local_path.touch()

    result = download.maybe_download(str(local_path))
    assert result == local_path

    with pytest.raises(FileNotFoundError):
        download.maybe_download("bogus")
