import uuid
import warnings
from dataclasses import dataclass, field, asdict
from typing import (
    Iterable,
    Dict,
    Optional,
    TYPE_CHECKING,
    Union,
    Tuple,
    List,
)

import numpy as np
import weaviate

from docarray import Document
from docarray.helper import dataclass_from_dict, filter_dict, _safe_cast_int
from docarray.array.storage.base.backend import BaseBackendMixin, TypeMap
from docarray.array.storage.registry import _REGISTRY

if TYPE_CHECKING:
    from docarray.typing import ArrayType, DocumentArraySourceType


@dataclass
class WeaviateConfig:
    """This class stores the config variables to initialize
    connection to the Weaviate server"""

    host: Optional[str] = field(default='localhost')
    port: Optional[int] = field(default=8080)
    protocol: Optional[str] = field(default='http')
    name: Optional[str] = None
    serialize_config: Dict = field(default_factory=dict)
    n_dim: Optional[int] = None  # deprecated, not used anymore since weaviate 1.10
    # vectorIndexConfig parameters
    ef: Optional[int] = None
    ef_construction: Optional[int] = None
    timeout_config: Optional[Tuple[int, int]] = None
    max_connections: Optional[int] = None
    dynamic_ef_min: Optional[int] = None
    dynamic_ef_max: Optional[int] = None
    dynamic_ef_factor: Optional[int] = None
    vector_cache_max_objects: Optional[int] = None
    flat_search_cutoff: Optional[int] = None
    cleanup_interval_seconds: Optional[int] = None
    skip: Optional[bool] = None
    columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    distance: Optional[str] = None


_banned_classname_chars = [
    '[',
    ' ',
    '"',
    '*',
    '\\',
    '<',
    '|',
    ',',
    '>',
    '/',
    '?',
    ']',
    '@',
    '.',
]


def _sanitize_class_name(name):
    """
    Sanitiazes class name if its not set correctly

    Parameters
    ----------
    results : string
        A sanitized string
    """

    new_name = name
    for char in _banned_classname_chars:
        new_name = new_name.replace(char, '')
    return new_name

def _check_batch_result(results: dict):
    """
    Check batch results for errors.

    Parameters
    ----------
    results : dict
        The Weaviate batch creation return value.
    """

    if results is not None:
        for result in results:
            if result.get('result', {}).get('errors', {}).get('error', None) is not None:
                if 'error' in result['result']['errors']:
                    raise ValueError(
                        'Weaviate throws an error',
                        result['result']['errors']['error'][0]['message']
                    )

class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    TYPE_MAP = {
        'str': TypeMap(type='string', converter=str),
        'float': TypeMap(type='number', converter=float),
        'int': TypeMap(type='int', converter=_safe_cast_int),
    }

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[WeaviateConfig, Dict]] = None,
        **kwargs,
    ):
        """Initialize weaviate storage.

        :param docs: the list of documents to initialize to
        :param config: the config object used to ininitialize connection to weaviate server
        :param kwargs: extra keyword arguments
        :raises ValueError: only one of name or docs can be used for initialization,
            raise an error if both are provided
        """

        if not config:
            config = WeaviateConfig()
        elif isinstance(config, dict):
            config = dataclass_from_dict(WeaviateConfig, config)

        self._serialize_config = config.serialize_config

        if config.name and config.name != config.name.capitalize():
            raise ValueError(
                'Weaviate class name has to be capitalized. '
                'Please capitalize when declaring the name field in config.'
            )

        self._persist = bool(config.name)

        self._client = weaviate.Client(
            f'{config.protocol}://{config.host}:{config.port}',
            timeout_config=config.timeout_config,
        )

        # Configure the Weaviate batch functionality
        self._client.batch(
            # Batch size starts with 1250 items per batch
            batch_size=1250,
            # If the batch size is too large, the Weaviate-client
            # updates the batch size dynamically
            dynamic=True,
            # Amount of times the clients aims to send a batch
            timeout_retries=3,
            creation_time=5,
            # Handle any error received from Weaviate
            callback=_check_batch_result,
        )

        self._config = config

        self._config.columns = self._normalize_columns(self._config.columns)

        self._schemas = self._load_or_create_weaviate_schema()

        _REGISTRY[self.__class__.__name__][self._class_name].append(self)

        super()._init_storage(_docs, **kwargs)

        # To align with Sqlite behavior; if `docs` is not `None` and table name
        # is provided, :class:`DocumentArraySqlite` will clear the existing
        # table and load the given `docs`
        if _docs is None:
            return
        elif isinstance(_docs, Iterable):
            self.clear()
            self.extend(_docs)
        else:
            self.clear()
            if isinstance(_docs, Document):
                self.append(_docs)

    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        if 'name' not in config_subindex:
            unique_name = _sanitize_class_name(
                config_joined['name'] + 'subindex' + subindex_name
            )
            config_joined['name'] = unique_name
        return config_joined

    def _get_schema_by_name(self, cls_name: str) -> Dict:
        """Return the schema dictionary object with the class name
        Content of the all dictionaries by this method are the same except the name
        of the weaviate's ``class``

        :param cls_name: the name of the schema/class in weaviate
        :return: the schema dictionary
        """

        hnsw_config = {
            'ef': self._config.ef,
            'efConstruction': self._config.ef_construction,
            'maxConnections': self._config.max_connections,
            'dynamicEfMin': self._config.dynamic_ef_min,
            'dynamicEfMax': self._config.dynamic_ef_max,
            'dynamicEfFactor': self._config.dynamic_ef_factor,
            'vectorCacheMaxObjects': self._config.vector_cache_max_objects,
            'flatSearchCutoff': self._config.flat_search_cutoff,
            'cleanupIntervalSeconds': self._config.cleanup_interval_seconds,
            'skip': self._config.skip,
            'distance': self._config.distance,
        }

        base_classes = {
            'classes': [
                {
                    'class': cls_name,
                    "vectorizer": "none",
                    'vectorIndexConfig': {'skip': False, **filter_dict(hnsw_config)},
                    'properties': [
                        {
                            'dataType': ['string'],
                            'name': '_docarray_id',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['blob'],
                            'name': 'blob',
                            'indexInverted': False,
                        },
                        {
                            'dataType': ['number[]'],
                            'name': 'tensor',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['text'],
                            'name': 'text',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['int'],
                            'name': 'granularity',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['int'],
                            'name': 'adjacency',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['string'],
                            'name': 'parent_id',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['number'],
                            'name': 'weight',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['string'],
                            'name': 'uri',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['string'],
                            'name': 'modality',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['string'],
                            'name': 'mime_type',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['number'],
                            'name': 'offset',
                            'indexInverted': True,
                        },
                        {
                            'dataType': ['number'],
                            'name': 'location',
                            'indexInverted': True,
                        },
                        {
                            'dataType': [cls_name],
                            'name': 'chunks'
                        },
                        {
                            'dataType': [cls_name],
                            'name': 'matches'
                        },
                    ],
                },
                {
                    'class': cls_name + 'Meta',
                    "vectorizer": "none",
                    'vectorIndexConfig': {'skip': True},
                    'properties': [
                        {
                            'dataType': ['string[]'],
                            'name': '_offset2ids',
                            'indexInverted': False,
                        },
                    ],
                },
            ]
        }
        for col, coltype in self._config.columns.items():
            new_property = {
                'dataType': [self._map_type(coltype)],
                'name': col,
                'indexInverted': True,
            }
            base_classes['classes'][0]['properties'].append(new_property)

        return base_classes

    def _load_or_create_weaviate_schema(self):
        """Create a new weaviate schema for this :class:`DocumentArrayWeaviate` object
        if not present in weaviate or if ``self._config.name`` is None. If ``self._config.name``
        is provided and not None and schema with the specified name exists in weaviate,
        then load the object with the given ``self._config.name``

        :return: the schemas of this :class`DocumentArrayWeaviate` object and its meta
        """
        if not self._config.name:
            warnings.warn("​You didn't set `config.name`, but it's better practice to set it. " + \
                "​This is because all Documents are stored as Weaviate Classes with their own vector spaces. " + \
                "For now, the class 'Document' is used.")
            self._config.name = 'Document'

        doc_schemas = self._get_schema_by_name(self._config.name)
        if self._client.schema.contains(doc_schemas):
            return doc_schemas

        self._client.schema.create(doc_schemas)
        return doc_schemas

    def _update_offset2ids_meta(self):
        """Update the offset2ids in weaviate the the current local version"""

        if self._offset2ids_wid is not None and self._client.data_object.exists(
            self._offset2ids_wid
        ):
            self._client.data_object.update(
                data_object={'_offset2ids': self._offset2ids.ids},
                class_name=self._meta_name,
                uuid=self._offset2ids_wid,
            )
        else:
            self._offset2ids_wid = str(uuid.uuid1())
            self._client.data_object.create(
                data_object={'_offset2ids': self._offset2ids.ids},
                class_name=self._meta_name,
                uuid=self._offset2ids_wid,
            )

    def _get_offset2ids_meta(self) -> Tuple[List, str]:
        """Return the offset2ids stored in weaviate along with the name of the schema/class
        in weaviate that stores meta information of this object

        :return: a tuple with first element as a list of offset2ids and second element
                 being name of weaviate class/schema of the meta object

        :raises ValueError: error is raised if meta class name is not defined
        """

        if not self._meta_name:
            raise ValueError('meta object is not defined')

        resp = (
            self._client.query.get(self._meta_name, ['_offset2ids', '_additional {id}'])
            .do()
            .get('data', {})
            .get('Get', {})
            .get(self._meta_name, [])
        )

        if not resp:
            return [], None
        elif len(resp) == 1:
            return resp[0]['_offset2ids'], resp[0]['_additional']['id']
        else:
            raise ValueError('received multiple meta copies which is invalid')

    @property
    def name(self):
        """An alias to _class_name that returns the id/name of the class
        in the weaviate of this :class:`DocumentArrayWeaviate`

        :return: name of weaviate class/schema of this :class:`DocumentArrayWeaviate`
        """

        return self._class_name

    @property
    def _class_name(self):
        """Return the name of the class in weaviate of this :class:`DocumentArrayWeaviate

        :return: name of weaviate class/schema of this :class:`DocumentArrayWeaviate`
        """

        if not self._schemas:
            return None
        return self._schemas['classes'][0]['class']

    @property
    def _meta_name(self):
        """Return the name of the class in weaviate that stores the meta information of
        this :class:`DocumentArrayWeaviate`

        :return: name of weaviate class/schema of class that stores the meta information
        """

        if not self._schemas:
            return None
        return self._schemas['classes'][1]['class']

    @property
    def _class_schema(self) -> Optional[Dict]:
        """Return the schema dictionary of this :class:`DocumentArrayWeaviate`'s weaviate schema

        :return: the dictionary representing this weaviate schema
        """

        if not self._schemas:
            return None
        return self._schemas['classes'][0]

    @property
    def _meta_schema(self):
        """Return the schema dictionary of this weaviate schema that stores this object's meta

        :return: the dictionary representing a meta object's weaviate schema
        """

        if not self._schemas and len(self._schemas) < 2:
            return None
        return self._schemas['classes'][1]

    def _doc2weaviate_create_payload(self, value: 'Document'):
        """Return the payload to store :class:`Document` into weaviate

        :param value: document to create a payload for
        :return: the payload dictionary
        """

        columns_dict = {key: val for [key, val] in self._config.columns}
        extra_columns = {
            col: self._map_column(value.tags.get(col), col_type)
            for col, col_type in self._config.columns.items()
        }

        data_object = {
            **extra_columns,
        }

        options_without_cross_refs = [ 'id',
                    'blob',
                    'tensor',
                    'text',
                    'granularity',
                    'adjacency',
                    'parent_id',
                    'weight',
                    'uri',
                    'modality',
                    'mime_type',
                    'offset',
                    'location',
                    'tags']

        options_with_cross_refs = ['chunks', 'matches']

        # set Weaviate object without cross references
        for opt in options_without_cross_refs:
            if getattr(value, opt):
                # filter out IDs
                if opt == 'id':
                    data_object['_docarray_id'] = getattr(value, opt)
                else:
                    data_object[opt] = getattr(value, opt)

        # set Weaviate object with cross references
        references = []
        for opt in options_with_cross_refs:
            if len(getattr(value, opt)) > 0:
                data_object[opt] = []
                for doc in getattr(value, opt):
                    references.append([
                        # from Weaviate uuid
                        self._map_id(getattr(value, 'id')),
                        # from class name
                        self._config.name,
                        # from property name
                        opt,
                        # to entity uuid
                        self._map_id(doc.id),
                        # to class name
                        self._config.name
                    ])

        return [dict(
                data_object=data_object,
                class_name=self._class_name,
                uuid=self._map_id(value.id),
                vector=self._map_embedding(value.embedding),
            ),
            references
        ]

    def _map_id(self, doc_id: str):
        """the function maps doc id to weaviate id

        :param doc_id: id of the document
        :return: weaviate object id
        """
        # appending class name to doc id to handle the case:
        # daw1 = DocumentArrayWeaviate([Document(id=str(i), text='hi') for i in range(3)])
        # daw2 = DocumentArrayWeaviate([Document(id=str(i), text='bye') for i in range(3)])
        # daw2[0, 'text'] == 'hi' # this will be False if we don't append class name
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id + self._class_name))

    def _map_embedding(self, embedding: 'ArrayType'):
        if embedding is not None:
            from docarray.math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()

            if len(embedding) == 1:
                embedding = [embedding[0]]
        else:
            embedding = None
        return embedding

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_client']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._client = weaviate.Client(
            f'{state["_config"].protocol}://{state["_config"].host}:{state["_config"].port}'
        )
        
