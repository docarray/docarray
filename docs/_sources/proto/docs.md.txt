# Protocol Documentation
<a name="top"></a>

## Table of Contents

- [docarray.proto](#docarray-proto)
    - [DenseNdArrayProto](#docarray-DenseNdArrayProto)
    - [DocumentArrayProto](#docarray-DocumentArrayProto)
    - [DocumentProto](#docarray-DocumentProto)
    - [DocumentProto.EvaluationsEntry](#docarray-DocumentProto-EvaluationsEntry)
    - [DocumentProto.ScoresEntry](#docarray-DocumentProto-ScoresEntry)
    - [NamedScoreProto](#docarray-NamedScoreProto)
    - [NdArrayProto](#docarray-NdArrayProto)
    - [SparseNdArrayProto](#docarray-SparseNdArrayProto)
  
- [Scalar Value Types](#scalar-value-types)



<a name="docarray-proto"></a>
<p align="right"><a href="#top">Top</a></p>

## docarray.proto



<a name="docarray-DenseNdArrayProto"></a>

### DenseNdArrayProto
Represents a (quantized) dense n-dim array


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| buffer | [bytes](#bytes) |  | the actual array data, in bytes |
| shape | [uint32](#uint32) | repeated | the shape (dimensions) of the array |
| dtype | [string](#string) |  | the data type of the array |






<a name="docarray-DocumentArrayProto"></a>

### DocumentArrayProto



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| docs | [DocumentProto](#docarray-DocumentProto) | repeated | a list of Documents |






<a name="docarray-DocumentProto"></a>

### DocumentProto
Represents a Document


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| id | [string](#string) |  | A hexdigest that represents a unique document ID |
| blob | [bytes](#bytes) |  | the raw binary content of this document, which often represents the original document when comes into jina |
| tensor | [NdArrayProto](#docarray-NdArrayProto) |  | the ndarray of the image/audio/video document |
| text | [string](#string) |  | a text document |
| granularity | [uint32](#uint32) |  | the depth of the recursive chunk structure |
| adjacency | [uint32](#uint32) |  | the width of the recursive match structure |
| parent_id | [string](#string) |  | the parent id from the previous granularity |
| weight | [float](#float) |  | The weight of this document |
| uri | [string](#string) |  | a uri of the document could be: a local file path, a remote url starts with http or https or data URI scheme |
| modality | [string](#string) |  | modality, an identifier to the modality this document belongs to. In the scope of multi/cross modal search |
| mime_type | [string](#string) |  | mime type of this document, for buffer content, this is required; for other contents, this can be guessed |
| offset | [float](#float) |  | the offset of the doc |
| location | [float](#float) | repeated | the position of the doc, could be start and end index of a string; could be x,y (top, left) coordinate of an image crop; could be timestamp of an audio clip |
| chunks | [DocumentProto](#docarray-DocumentProto) | repeated | list of the sub-documents of this document (recursive structure) |
| matches | [DocumentProto](#docarray-DocumentProto) | repeated | the matched documents on the same level (recursive structure) |
| embedding | [NdArrayProto](#docarray-NdArrayProto) |  | the embedding of this document |
| tags | [google.protobuf.Struct](#google-protobuf-Struct) |  | a structured data value, consisting of field which map to dynamically typed values. |
| scores | [DocumentProto.ScoresEntry](#docarray-DocumentProto-ScoresEntry) | repeated | Scores performed on the document, each element corresponds to a metric |
| evaluations | [DocumentProto.EvaluationsEntry](#docarray-DocumentProto-EvaluationsEntry) | repeated | Evaluations performed on the document, each element corresponds to a metric |






<a name="docarray-DocumentProto-EvaluationsEntry"></a>

### DocumentProto.EvaluationsEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [NamedScoreProto](#docarray-NamedScoreProto) |  |  |






<a name="docarray-DocumentProto-ScoresEntry"></a>

### DocumentProto.ScoresEntry



| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| key | [string](#string) |  |  |
| value | [NamedScoreProto](#docarray-NamedScoreProto) |  |  |






<a name="docarray-NamedScoreProto"></a>

### NamedScoreProto
Represents the relevance model to `ref_id`


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| value | [float](#float) |  | value |
| op_name | [string](#string) |  | the name of the operator/score function |
| description | [string](#string) |  | text description of the score |
| ref_id | [string](#string) |  | the score is computed between doc `id` and `ref_id` |






<a name="docarray-NdArrayProto"></a>

### NdArrayProto
Represents a general n-dim array, can be either dense or sparse


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| dense | [DenseNdArrayProto](#docarray-DenseNdArrayProto) |  | dense representation of the ndarray |
| sparse | [SparseNdArrayProto](#docarray-SparseNdArrayProto) |  | sparse representation of the ndarray |
| cls_name | [string](#string) |  | the name of the ndarray class |
| parameters | [google.protobuf.Struct](#google-protobuf-Struct) |  |  |






<a name="docarray-SparseNdArrayProto"></a>

### SparseNdArrayProto
Represents a sparse ndarray


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| indices | [DenseNdArrayProto](#docarray-DenseNdArrayProto) |  | A 2-D int64 tensor of shape [N, ndims], which specifies the indices of the elements in the sparse tensor that contain nonzero values (elements are zero-indexed) |
| values | [DenseNdArrayProto](#docarray-DenseNdArrayProto) |  | A 1-D tensor of any type and shape [N], which supplies the values for each element in indices. |
| shape | [uint32](#uint32) | repeated | A 1-D int64 tensor of shape [ndims], which specifies the shape of the sparse tensor. |





 

 

 

 



## Scalar Value Types

| .proto Type | Notes | C++ | Java | Python | Go | C# | PHP | Ruby |
| ----------- | ----- | --- | ---- | ------ | -- | -- | --- | ---- |
| <a name="double" /> double |  | double | double | float | float64 | double | float | Float |
| <a name="float" /> float |  | float | float | float | float32 | float | float | Float |
| <a name="int32" /> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="int64" /> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="uint32" /> uint32 | Uses variable-length encoding. | uint32 | int | int/long | uint32 | uint | integer | Bignum or Fixnum (as required) |
| <a name="uint64" /> uint64 | Uses variable-length encoding. | uint64 | long | int/long | uint64 | ulong | integer/string | Bignum or Fixnum (as required) |
| <a name="sint32" /> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="sint64" /> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="fixed32" /> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int | int | uint32 | uint | integer | Bignum or Fixnum (as required) |
| <a name="fixed64" /> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | long | int/long | uint64 | ulong | integer/string | Bignum |
| <a name="sfixed32" /> sfixed32 | Always four bytes. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="sfixed64" /> sfixed64 | Always eight bytes. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="bool" /> bool |  | bool | boolean | boolean | bool | bool | boolean | TrueClass/FalseClass |
| <a name="string" /> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | String | str/unicode | string | string | string | String (UTF-8) |
| <a name="bytes" /> bytes | May contain any arbitrary sequence of bytes. | string | ByteString | str | []byte | ByteString | string | String (ASCII-8BIT) |

