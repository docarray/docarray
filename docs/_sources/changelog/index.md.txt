# Changelog

DocArray follows semantic versioning. However, before the project reach 1.0.0, any breaking change will only bump the minor version.  An automated release note is [generated on every release](https://github.com/jina-ai/docarray/releases). The release note includes features, bugs, refactorings etc. 

This chapter only tracks the most important breaking changes and explain the rationale behind them.

## 0.3.0: change on the default JSON/dict serialization strategy

This change is a breaking change and is not back-compatible.

Document/DocumentArray now favors schema-ed JSON over "unschema-ed" JSON in both JSON & dict IO interfaces. Specifically, 0.3.0 introduces `protocol='jsonschema'` (as default) and `protocol='protobuf'` to allow user to control the serialization behavior.

Migration guide:

- Read the docs: {ref}`doc-json`.
- If you are using `.to_dict()`, `.to_json()`, `.from_dict()`, `.from_json()` at Document/DocumentArray level, please be aware the change of JSON output.
- If you want to stick to old Protobuf-based JSON (not recommended, as it is "unschema-ed"), use `.to_json(protocol='protobuf')` and `.from_json(protocol='protobuf')`.
- Fine-grained controls can be archived by passing extra key-value args as described in {ref}`doc-json`.


## 0.2.0: change on the content field name

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

