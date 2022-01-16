# Changelog

DocArray follows semantic versioning. However, before the project reach 1.0.0, any breaking change will only bump the minor version.  An automated release note is [generated on every release](https://github.com/jina-ai/docarray/releases). The release note includes features, bugs, refactorings etc. 

This chapter only tracks the most important breaking changes and explain the rationale behind them.

## 0.2.0: change the content field name

**This change is a breaking change and is not back-compatible.**

The Document schema is changed as follows:

| 0.1.x     | 0.2       | Semantic                                             |
|-----------|-----------|------------------------------------------------------|
| `.blob`   | `.tensor` | To represent the ndarray of a Document               |
| `.buffer` | `.blob`   | To represent the binary representation of a Document |

This changed is made based on the word "BLOB" is a well-acknowledged as "binary large object" in the database field. It is a more natural wording for representing binary and less natural for representing `ndarray`. Previously, only Caffee used `blob` to represent `ndarray`.

Unifying the terminology also avoids confusion when integrate DocArray into some databases.

All fluent interfaces of `Document` are also changed accordingly.

Here is a short migration guide for 0.1.x users:

| Old                                      | New                                 | Remark                                                                         |
|------------------------------------------|-------------------------------------|--------------------------------------------------------------------------------|
| Document.blob                            | Document.tensor                     |                                                                                |
| Document.buffer                          | Document.blob                       |                                                                                |
| DocumentArray.blobs, `da[:, 'blob']`     | Document.tensors, `da[:, 'tensor']` |                                                                                |
| DocumentArray.buffers, `da[:, 'buffer']` | Document.blobs, `da[:, 'blob']`     |                                                                                |
| Document.*_blob_*                        | Document.*_tensor_*                 | Apply to [all functions in here](../fundamentals/document/fluent-interface.md) |
| Document.*_buffer_*                      | Document.*_blob_*                   | Apply to [all functions in here](../fundamentals/document/fluent-interface.md) |

JSON Schema needs to be re-generated {ref}`by following this<schema-gen>`.

