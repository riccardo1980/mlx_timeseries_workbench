import pytest
from timeseries_autoencoder import data


@pytest.fixture
def ds():
    return data.Dataset('ECG5000', 'https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip', 'ECG5000.zip')


def test_ECG5000_download_file_exists(mocker, ds):

    exists_mock = mocker.patch('os.path.exists')
    makedirs_mock = mocker.patch('os.makedirs')
    requests_get_mock = mocker.patch('requests.get')
    open_mock = mocker.patch('builtins.open')
    
    exists_mock.return_value = True

    data.dataset_download(ds, force=False)

    exists_mock.assert_called_once_with(ds.filename)
    makedirs_mock.assert_not_called()
    requests_get_mock.assert_not_called()
    open_mock.assert_not_called()

def test_ECG5000_download_file_not_exist(mocker, ds):

    exists_mock = mocker.patch('os.path.exists')
    makedirs_mock = mocker.patch('os.makedirs')
    requests_get_mock = mocker.patch('requests.get')
    open_mock = mocker.patch('builtins.open', mocker.mock_open())
    
    exists_mock.return_value = False
    makedirs_mock.return_value = None
    requests_get_mock.return_value = mocker.MagicMock()
    requests_get_mock.return_value.iter_content.return_value = [1]

    data.dataset_download(ds, force=False)

    exists_mock.assert_called_once_with(ds.filename)
    makedirs_mock.assert_called_once_with(data.DATA_CACHE_BASE_FOLDER, exist_ok=True)
    requests_get_mock.assert_called_once_with(ds.link, stream=True)
    open_mock.assert_called_once_with(ds.filename, 'wb')
    open_mock.return_value.__enter__.assert_called_once()
    open_mock.return_value.write.assert_called_once_with(1)
    open_mock.return_value.__exit__.assert_called_once_with(None, None, None)

def test_ECG5000_download_force_file_exists(mocker, ds):

    exists_mock = mocker.patch('os.path.exists')
    makedirs_mock = mocker.patch('os.makedirs')
    requests_get_mock = mocker.patch('requests.get')
    open_mock = mocker.patch('builtins.open', mocker.mock_open())
    
    exists_mock.return_value = True
    makedirs_mock.return_value = None
    requests_get_mock.return_value = mocker.MagicMock()
    requests_get_mock.return_value.iter_content.return_value = [1]

    data.dataset_download(ds, force=True)

    exists_mock.assert_called_once_with(ds.filename)
    makedirs_mock.assert_called_once_with(data.DATA_CACHE_BASE_FOLDER, exist_ok=True)
    requests_get_mock.assert_called_once_with(ds.link, stream=True)
    open_mock.assert_called_once_with(ds.filename, 'wb')
    open_mock.return_value.__enter__.assert_called_once()
    open_mock.return_value.write.assert_called_once_with(1)
    open_mock.return_value.__exit__.assert_called_once_with(None, None, None)

def test_ECG5000_download_force_file_not_exist(mocker, ds):

    exists_mock = mocker.patch('os.path.exists')
    makedirs_mock = mocker.patch('os.makedirs')
    requests_get_mock = mocker.patch('requests.get')
    open_mock = mocker.patch('builtins.open', mocker.mock_open())
    
    exists_mock.return_value = False
    makedirs_mock.return_value = None
    requests_get_mock.return_value = mocker.MagicMock()
    requests_get_mock.return_value.iter_content.return_value = [1]

    data.dataset_download(ds, force=True)

    exists_mock.assert_called_once_with(ds.filename)
    makedirs_mock.assert_called_once_with(data.DATA_CACHE_BASE_FOLDER, exist_ok=True)
    requests_get_mock.assert_called_once_with(ds.link, stream=True)
    open_mock.assert_called_once_with(ds.filename, 'wb')
    open_mock.return_value.__enter__.assert_called_once()
    open_mock.return_value.write.assert_called_once_with(1)
    open_mock.return_value.__exit__.assert_called_once_with(None, None, None)
