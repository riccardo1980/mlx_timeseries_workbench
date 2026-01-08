import pytest
import requests
from pytest_mock import MockerFixture

from mlx_timeseries_workbench import data


@pytest.mark.parametrize(
    "filename, link, exists, force, got",
    [
        ("filename", "link", False, False, "download"),
        ("filename", "link", False, True, "download"),
        ("filename", "link", True, False, "skip"),
        ("filename", "link", True, True, "download"),
    ],
)
def test_maybe_download(
    mocker: MockerFixture,
    filename: str,
    link: str,
    exists: bool,
    force: bool,
    got: str,
) -> None:
    exists_mock = mocker.patch("os.path.exists")
    requests_get_mock = mocker.patch("requests.get")
    requests_get_mock.return_value.iter_content.return_value = [b"chunk"]
    open_mock = mocker.patch("mlx_timeseries_workbench.data.open", mocker.mock_open())

    exists_mock.return_value = exists
    data.__maybe_download(link, filename, force=force)

    exists_mock.assert_called_once_with(filename)
    if got == "skip":
        requests_get_mock.assert_not_called()
        open_mock.assert_not_called()
        open_mock.return_value.__enter__.assert_not_called()
        open_mock.return_value.write.assert_not_called()
        open_mock.return_value.__exit__.assert_not_called()

    if got == "download":
        requests_get_mock.assert_called_once_with(link)
        requests_get_mock.return_value.iter_content.assert_called_once()
        open_mock.assert_called_once_with(filename, "wb")
        open_mock.return_value.__enter__.assert_called_once()
        open_mock.return_value.write.assert_called_once_with(b"chunk")
        open_mock.return_value.__exit__.assert_called_once_with(None, None, None)


def test_maybe_download_network_error(mocker: MockerFixture) -> None:
    mocker.patch("os.path.exists", return_value=False)
    mocker.patch("requests.get", side_effect=requests.exceptions.ConnectionError)

    with pytest.raises(requests.exceptions.ConnectionError):
        data.__maybe_download("link", "filename")
