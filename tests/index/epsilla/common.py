epsilla_config = {
    "protocol": 'http',
    "host": 'localhost',
    "port": 8888,
    "is_self_hosted": True,
    "db_path": "/epsilla",
    "db_name": "tony_doc_array_test",
}


def index_len(index, max_len=20):
    return len(index.filter("", limit=max_len))
