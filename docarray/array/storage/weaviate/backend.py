import itertools
import uuid
from dataclasses import dataclass, field
from typing import (
    Generator,
    Iterator,
    Dict,
    Sequence,
    Optional,
    TYPE_CHECKING,
    Union,
    Tuple,
    List,
)

import scipy.sparse
import weaviate

from ..base.backend import BaseBackendMixin
from .... import Document

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


@dataclass
class WeaviateConfig:
    """This class stores the config variables to initialize
    connection to the Weaviate server"""

    client: Optional[Union[str, weaviate.Client]] = None
    n_dim: Optional[int] = None
    name: Optional[str] = None
    serialize_config: Dict = field(default_factory=dict)


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend. """

    def _init_storage(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[WeaviateConfig] = None,
        **kwargs
    ):
        """Initialize weaviate storage.

        :param docs: the list of documents to initialize to
        :param config: the config object used to ininitialize connection to weaviate server
        :param kwargs: extra keyword arguments
        :raises ValueError: only one of name or docs can be used for initialization,
            raise an error if both are provided
        """

        from ... import DocumentArray

        self._schemas = None

        if not config:
            config = WeaviateConfig()

        self.n_dim = config.n_dim or 1
        self.serialize_config = config.serialize_config

        import weaviate

        if config.client is None:
            config.client = 'http://localhost:8080'

        if isinstance(config.client, str):
            self._client = weaviate.Client(config.client)
        else:
            self._client = config.client

        if config.name is not None and docs is not None:
            raise ValueError(
                'only one of name or docs can be provided for initialization'
            )
        self._config = config

        self._schemas = self._load_or_create_weaviate_schema()
        self._offset2ids, self._offset2ids_wid = self._get_offset2ids_meta()

        if docs is None and config.name:
            return

        # To align with Sqlite behavior; if `docs` is not `None` and table name
        # is provided, :class:`DocumentArraySqlite` will clear the existing
        # table and load the given `docs`
        self.clear()
        if isinstance(
            docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            self.extend(docs)
        else:
            if isinstance(docs, Document):
                self.append(docs)

    def _get_weaviate_class_name(self) -> str:
        """Generate the class/schema name using the ``uuid1`` module with some
        formatting to tailor to weaviate class name convention

        :return: string representing the name of  weaviate class/schema name of
            this :class:`DocumentArrayWeaviate` object
        """
        return ''.join([i for i in uuid.uuid1().hex if not i.isdigit()]).capitalize()

    def _get_schema_by_name(self, cls_name: str) -> Dict:
        """Return the schema dictionary object with the class name
        Content of the all dictionaries by this method are the same except the name
        of the weaviate's ``class``

        :param cls_name: the name of the schema/class in weaviate
        :return: the schema dictionary
        """
        # TODO: ideally we should only use one schema. this will allow us to deal with
        # consistency better
        return {
            'classes': [
                {
                    'class': cls_name,
                    "vectorizer": "none",
                    # TODO: this skips checking embedding dimension but might not
                    # work if want to leverage weaviate for vector search
                    'vectorIndexConfig': {'skip': True},
                    'properties': [
                        {
                            'dataType': ['blob'],
                            'name': '_serialized',
                            'indexInverted': False,
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

    def _load_or_create_weaviate_schema(self):
        """Create a new weaviate schema for this :class:`DocumentArrayWeaviate` object
        if not present in weaviate or if ``cls_name`` not provided, else if ``cls_name`` is provided
        load the object with the given ``cls_name``

        :param cls_name: if provided, load this :class:`DocumentArrayWeaviate` object with
            the data stored in weaviate. If ``cls_name`` is ``None``, the create a schema in weaviate
            with a newly generated class name.
        :return: the schemas of this :class`DocumentArrayWeaviate` object and its meta
        """
        if not self._config.name:
            name_candidate = self._get_weaviate_class_name()
            doc_schemas = self._get_schema_by_name(name_candidate)
            while self._client.schema.contains(doc_schemas):
                name_candidate = self._get_weaviate_class_name()
                doc_schemas = self._get_schema_by_name(name_candidate)
            self._client.schema.create(doc_schemas)
            self._config.name = name_candidate
            return doc_schemas

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
                data_object={'_offset2ids': self._offset2ids},
                class_name=self._meta_name,
                uuid=self._offset2ids_wid,
            )
        else:
            self._offset2ids_wid = str(uuid.uuid1())
            self._client.data_object.create(
                data_object={'_offset2ids': self._offset2ids},
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
        # TODO: remove this after we combine the meta info to the DocumentArray class
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
        if value.embedding is None:
            embedding = [0] * self.n_dim
        elif isinstance(value.embedding, scipy.sparse.spmatrix):
            embedding = value.embedding.toarray()
        else:
            from ....math.ndarray import to_numpy_array

            embedding = to_numpy_array(value.embedding)

        return dict(
            data_object={'_serialized': value.to_base64(**self.serialize_config)},
            class_name=self._class_name,
            uuid=self.wmap(value.id),
            vector=embedding,
        )

    def wmap(self, doc_id: str):
        """the function maps doc id to weaviate id

        :param doc_id: id of the document
        :return: weaviate object id
        """
        # appending class name to doc id to handle the case:
        # daw1 = DocumentArrayWeaviate([Document(id=str(i), text='hi') for i in range(3)])
        # daw2 = DocumentArrayWeaviate([Document(id=str(i), text='bye') for i in range(3)])
        # daw2[0, 'text'] == 'hi' # this will be False if we don't append class name
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id + self._class_name))

    def _get_storage_infos(self) -> Dict:
        storage_infos = super()._get_storage_infos()
        return {
            'Backend': 'Weaviate (www.semi.technology/developers/weaviate)',
            'Hostname': self._config.client,
            'Schema Name': self._config.name,
            'Serialization Protocol': self._config.serialize_config.get('protocol'),
            **storage_infos,
        }
