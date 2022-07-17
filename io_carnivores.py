import bpy
import struct

bl_info = {
    "name": "Carnivores Export (3DF,VTL)",
    "author": "Ithamar R. Adema",
    "version": (1, 0, 0),
    "blender": (2, 82, 0),
    "description": "Export plugin for classic Carnivores formats 3DF (base model), VTL (animations)",
    "category": "Import-Export",
}

def mesh_triangulate(me):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()

def export_3df(context, filepath, mat):
    depsgraph = context.evaluated_depsgraph_get()

    # get selected object and create mesh
    obj = context.selected_objects[0]
    obj_for_convert = obj.evaluated_get(depsgraph)
    mesh = obj_for_convert.to_mesh()

    # Triangulate mesh
    mesh_triangulate(mesh)
    mesh.transform(mat)

    # UV layer
    uv_layer = mesh.uv_layers.active.data

    face_count = len(mesh.polygons)

    f = open(filepath, 'wb')

    # Write Header
    f.write(struct.pack("<LLLL", len(mesh.vertices), face_count, 0, 0)) # bones, texturesize

    # Write faces
    for poly in mesh.polygons:
        # Start of poly loop
        start = poly.loop_start

        f.write(
            struct.pack("<LLLLLLLLLHHLLLxxxxxxxxxxxx",
            mesh.loops[start + 0].vertex_index,
            mesh.loops[start + 1].vertex_index,
            mesh.loops[start + 2].vertex_index,
            int(uv_layer[start + 0].uv.x * 255),
            int(uv_layer[start + 1].uv.x * 255),
            int(uv_layer[start + 2].uv.x * 255),
            int(uv_layer[start + 0].uv.y * 255),
            int(uv_layer[start + 1].uv.y * 255),
            int(uv_layer[start + 2].uv.y * 255),
            0, # flags
            0, # dmask
            0, # distant
            0, # next
            0, # group
            )
        )

    # Write Vertices
    for v in mesh.vertices:
        f.write(struct.pack("<fffHH", v.co.x, v.co.y, v.co.z, 0, 0)) # bone and hide

    f.close()

    return {'FINISHED'}

def export_vtl(context, filepath, frame_step, mat):
    # Get copy of active mesh

    scene = context.scene
    orig_frame = scene.frame_current
    scene_frames = range(scene.frame_start, scene.frame_end + 1, frame_step)

    fps = scene.render.fps / frame_step

    f = open(filepath, 'wb')
    f.write(struct.pack("<LLL", 0, int(fps), len(scene_frames))) # VCount, aniKPS, framesCount

    vcount = 0

    depsgraph = context.evaluated_depsgraph_get()

    for frame in scene_frames:
        scene.frame_set(frame, subframe=0.0)

        # get selected object and create mesh
        obj = context.selected_objects[0]
        obj_for_convert = obj.evaluated_get(depsgraph)
        mesh = obj_for_convert.to_mesh()

        # Triangulate mesh
        mesh_triangulate(mesh)
        mesh.transform(mat)

        # Write Vertices
        vcount = len(mesh.vertices)
        for v in mesh.vertices:
            f.write(struct.pack("<hhh",
            int(v.co.x * 16),
            int(v.co.y * 16),
            int(v.co.z * 16),
            ))

        obj_for_convert.to_mesh_clear()

    # Now patch in vertex count
    f.seek(0)
    f.write(struct.pack("<L", vcount))

    # ... and we're done!
    f.close()

    return {'FINISHED'}

# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper, orientation_helper, axis_conversion
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty
from bpy.types import Operator

@orientation_helper(axis_forward='-Z', axis_up='Y')
class ExportVTL(Operator, ExportHelper):
    """Export classic Carnivores VTL animation format"""
    bl_idname = "export_vtl.some_data"
    bl_label = "Export VTL (Carnivores)"
    #bl_options = {'PRESET'}

    # ExportHelper mixin class uses this
    filename_ext = ".vtl"

    filter_glob: StringProperty(
        default="*.vtl",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.

    #frame_step: IntProperty(
    #    name="Frame Step",
    #    description="Export every Nth frame (default=1=all)",
    #    hard_min=1, hard_max=1000,
    #    soft_min=1, soft_max=1000,
    #    default=1,
    #    step=1,
    #)

    def execute(self, context):
        return export_vtl(context, self.filepath, 1, axis_conversion(to_forward=self.axis_forward,to_up=self.axis_up).to_4x4())


@orientation_helper(axis_forward='-Z', axis_up='Y')
class Export3DF(Operator, ExportHelper):
    """Export classic Carnivores 3DF model format"""
    bl_idname = "export_3df.some_data"
    bl_label = "Export 3DF (Carnivores)"
    #bl_options = {'PRESET'}

    # ExportHelper mixin class uses this
    filename_ext = ".3df"

    filter_glob: StringProperty(
        default="*.3df",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    def execute(self, context):
        return export_3df(context, self.filepath, axis_conversion(to_forward=self.axis_forward,to_up=self.axis_up).to_4x4())


# Only needed if you want to add into a dynamic menu
def menu_func_export(self, context):
    self.layout.operator(Export3DF.bl_idname, text="3DF (Carnivores)")
    self.layout.operator(ExportVTL.bl_idname, text="VTL (Carnivores)")

def register():
    bpy.utils.register_class(Export3DF)
    bpy.utils.register_class(ExportVTL)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.utils.unregister_class(Export3DF)
    bpy.utils.unregister_class(ExportVTL)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
