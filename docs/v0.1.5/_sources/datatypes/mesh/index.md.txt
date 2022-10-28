(mesh-type)=
# {octicon}`package` 3D Mesh

````{tip}
This feature requires `trimesh`. You can install it via `pip install "docarray[full]".` 
````

A 3D mesh is the structural build of a 3D model consisting of polygons. Most 3D meshes are created via professional software packages, such as commercial suites like Unity, or the free open source Blender 3D.

## Point cloud

Point cloud is a representation of a 3D mesh. It is made by repeated and uniformly sampling points within the 3D body. Comparing to the mesh representation, point cloud is a fixed size ndarray and hence easier for deep learning algorithms to handle. In DocArray, you can simply load a 3D mesh and convert it into a point cloud via:

```python
from docarray import Document
doc = Document(uri='viking.glb').load_uri_to_point_cloud_blob(1000)

print(doc.blob.shape)
```

```text
(1000, 3)
```

The following pictures depict a 3D mesh and a point cloud with 1000 samples from that 3D mesh. 

```{figure} 3dmesh-man.gif
:width: 50%
```

```{figure} pointcloud-man.gif
:width: 50%
```
