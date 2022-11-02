(mesh-type)=
# {octicon}`package` 3D Mesh

````{tip}
This feature requires `trimesh`. You can install it via `pip install "docarray[full]".` 
````

A 3D mesh is the structural build of a 3D model consisting of polygons. Most 3D meshes are created via professional software packages, such as commercial suites like Unity, or the free open source Blender 3D.

## Vertices and faces representation 

A 3D mesh can be represented by its vertices and faces. Vertices are points in a 3D space, represented as a tensor of shape (n_points, 3). Faces are triangular surfaces that can be defined by three points in 3D space, corresponding to the three vertices of a triangle. Faces can be represented as a tensor of shape (n_faces, 3). Each number in that tensor refers to an index of a vertex in the tensor of vertices.

Additionally, a 3D mesh can contain textural information. One option to represent the texture is with an image and the corresponding uv-mapping. The uv-mapping is a tensor of shape (n_vertices, 2). Each row describes the mapping of a corresponding vertex onto the uv-image.

In DocArray, you can load a mesh and save its vertices, faces and textural information to a Document's `.chunks` as follows:

```python
from docarray import Document

doc = Document(uri='viking.glb').load_uri_to_vertices_and_faces()

doc.summary()
```

```text
 <Document ('id', 'chunks') at 7f907d786d6c11ec840a1e008a366d49>
    └─ chunks
          ├─ <Document ('id', 'parent_id', 'granularity', 'tensor', 'tags') at 7f907ab26d6c11ec840a1e008a366d49>
          └─ <Document ('id', 'parent_id', 'granularity', 'tensor', 'tags') at 7f907c106d6c11ec840a1e008a366d49>
          └─ <Document ('id', 'parent_id', 'granularity', 'tensor', 'tags') at 7f907f106d6c11ec840a1e008a366d49>
          └─ <Document ('id', 'parent_id', 'granularity', 'tensor', 'tags') at 7f907k106d6c11ec840a1e008a366d49>
```

This stores the vertices, faces, uv-image and uv-mapping in `.tensor` of four separate sub-Documents in a Document's `.chunks`. Each sub-Document has a name assigned to it ('vertices', 'faces', 'uv_image' or 'uv_mapping'), which is saved in `.tags`:

```python
for chunk in doc.chunks:
    print(f'chunk.tags = {chunk.tags}')
```

```text
chunk.tags = {'name': 'vertices'}
chunk.tags = {'name': 'faces'}
chunk.tags = {'name': 'uv_image'}
chunk.tags = {'name': 'uv_mapping'}
```

The following picture depicts a 3D mesh:

```{figure} 3dmesh-man.gif
:width: 50%
```

## Point cloud representation

A point cloud is a representation of a 3D mesh. It is made by repeatedly and uniformly sampling points within the 3D body. Compared to the mesh representation, the point cloud is a fixed size ndarray and hence easier for deep learning algorithms to handle. In DocArray, you can simply load a 3D mesh and convert it into a point cloud of size `samples` via:

```python
from docarray import Document

doc = Document(uri='viking.glb').load_uri_to_point_cloud_tensor(samples=1000)

print(doc.tensor.shape)
```

```text
(1000, 3)
```

The following picture depicts a point cloud with 1000 samples from the previously depicted 3D mesh.

```{figure} pointcloud-man.gif
:width: 50%
```
