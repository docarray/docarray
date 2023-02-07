from jina import DocumentArray
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from datetime import datetime
now = datetime.now()
print(f'start: {now.strftime("%H:%M:%S")}')

# should create the da from existing ES index
# NOTE: has yet to finish
da = DocumentArray(
	storage='elasticsearch',
	config={
		'index_name': 'dataset100m',
		'n_dim': 200
	}
)
print(datetime.now() - now)
print(len(da))
