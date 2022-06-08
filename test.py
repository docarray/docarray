from docarray import DocumentArray, Document
import numpy as np

da = DocumentArray(storage='elasticsearch', config={'n_dim': 10, 'index_name': 'old_stuff',
                                                    })
da.extend(
    [
        Document(id='1', embedding=np.random.rand(1, 10)),
        Document(id='2', embedding=np.random.rand(1, 10)),
        Document(id='3', embedding=np.random.rand(1, 10)),
    ]
)

id = ['1', '2']

pizza_docs = da.find(query={
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": {"ids": {"values": id}}
                        },
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query, 'embedding') + 1.0",
                        "params": {"query": np.random.rand(10)},
                    },
                }
        })
print(pizza_docs)


# da = DocumentArray(storage='elasticsearch', config={'n_dim': 10, 'distance': 'l2_norm', 'index_text': True})
#
# da.extend(
#     [
#         Document(id='1', text='Person eating'),
#         Document(id='2', text='Person eating pizza'),
#         Document(id='3', text='Pizza restaurant'),
#     ]
# )
#
# id = ['1', '2', '3']
# pizza_docs = da.find(query={
#                             "bool": {
#                                 "must": [
#                                             {"match": {
#                                                 "text": 'pizza',
#                                                 }
#                                             }
#                                 ],
#                                 "filter": {
#                                     "ids": {"values": id}
#                                 }
#                             }
#         })
# print(pizza_docs[:, 'text'])
