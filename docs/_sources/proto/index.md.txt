```{include} docs.md
```

## Rebuild Protobuf

To rebuild `docarray.proto` :

```bash
cd docarray
docker run -v $(pwd)/proto:/jina/proto jinaai/protogen
```