"""
Headless Blender mesh cleanup and GLB export.

Invoked as:
  blender --background --python blender_export.py -- <input_glb> <output_glb> [decimate_ratio] [max_triangles]
"""

from __future__ import annotations

import sys


def _parse_args() -> tuple[str, str, float, int]:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit(
            "Usage: blender --background --python blender_export.py -- "
            "<in.glb> <out.glb> [ratio] [max_tris]"
        )
    args = argv[argv.index("--") + 1 :]
    if len(args) < 2:
        raise SystemExit("Missing input_glb and output_glb paths")
    ratio = float(args[2]) if len(args) > 2 else 0.5
    max_tris = int(args[3]) if len(args) > 3 else 100_000
    return args[0], args[1], ratio, max_tris


def main() -> None:
    import bpy  # type: ignore[import-untyped]

    input_glb, output_glb, decimate_ratio, max_triangles = _parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)

    if bpy.ops.import_scene.gltf(filepath=input_glb) != {"FINISHED"}:
        raise RuntimeError(f"Blender failed to import GLB: {input_glb}")

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh objects found after GLB import")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
    bpy.ops.object.mode_set(mode="OBJECT")

    tri_count = sum(
        len(obj.data.polygons) for obj in bpy.context.scene.objects if obj.type == "MESH"
    )
    if tri_count > max_triangles:
        for obj in bpy.context.scene.objects:
            if obj.type == "MESH":
                mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
                mod.ratio = min(decimate_ratio, max_triangles / max(tri_count, 1))
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=output_glb,
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )


if __name__ == "__main__":
    main()
