from jina import DocumentArray, Document
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from datetime import datetime
data = np.load('/Backup/data/benchmark/dataset_100M.npy')
print(f'data loaded, shape: {data.shape}')

da = DocumentArray(
	storage='elasticsearch',
	config={
		'index_name': 'dataset100m',
		'n_dim': 200
	}
)
now = datetime.now()
print(f'start time: {now}')
with da:
	da.extend(
		[Document(embedding=data[i]) for i in range(len(data))]
	)
later = datetime.now()
print(f'finish time: {later}')
print(f'time in sec: {(later - now).total_seconds()}')
