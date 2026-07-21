from unittest.mock import MagicMock, patch

from pymongo.driver_info import DriverInfo


def test_connect_passes_driver_info():
    """_connect_to_mongodb_atlas passes DriverInfo(name='DocArray') to MongoClient."""
    from docarray.index.backends.mongodb_atlas import MongoDBAtlasDocumentIndex

    with patch("docarray.index.backends.mongodb_atlas.MongoClient") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        MongoDBAtlasDocumentIndex._connect_to_mongodb_atlas("mongodb://localhost")

        mock_client_cls.assert_called_once()
        _, kwargs = mock_client_cls.call_args
        assert isinstance(kwargs.get("driver"), DriverInfo)
        assert kwargs["driver"].name == "DocArray"
