======================
:fab:`python` Python API
======================

This section includes the API documentation from the `docarray` codebase, as extracted from the `docstrings <https://peps.python.org/pep-0257/>`_ in the code.

For further details, please refer to the full :ref:`user guide <document>`.


:mod:`docarray` - Document and DocumentArray
--------------------

.. currentmodule:: docarray

.. autosummary::
   :nosignatures:
   :template: class.rst

   document.Document
   array.document.DocumentArray
   array.chunk.ChunkArray
   array.match.MatchArray



:mod:`docarray.dataclasses` - Dataclass
--------------------

.. currentmodule:: docarray.dataclasses

.. autosummary::
   :nosignatures:
   :template: class.rst

   types.dataclass
   types.is_multimodal
   types.field
   
   
:mod:`docarray.array` - Document stores
--------------------

.. currentmodule:: docarray.array

.. autosummary::
   :nosignatures:
   :template: class.rst

   memory.DocumentArrayInMemory
   sqlite.DocumentArraySqlite
   annlite.DocumentArrayAnnlite
   weaviate.DocumentArrayWeaviate
   qdrant.DocumentArrayQdrant
   elastic.DocumentArrayElastic



