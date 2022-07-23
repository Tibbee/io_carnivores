import bpy
import struct

bl_info = {
    "name": "Carnivores Model/Animation Export (3DF,VTL)",
    "author": "Ithamar R. Adema",
    "version": (1, 0, 2),
    "blender": (2, 82, 0),
    "description": "Export plugin for classic Carnivores formats 3DF (base model), VTL (animations)",
    "category": "Import-Export",
}

def prepare_export(context, operator):
    # Exit edit mode before exporting, so current object states are exported properly.
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    # Determine object to export
    obj = context.active_object
    if obj != None:
        if obj.type != 'MESH':
            # If an armature was selected, find corresponding mesh
            if obj.type == 'ARMATURE':
                for o in obj.children:
                    if o.type == 'MESH':
                        obj = o
                        break
                # If we didn't find a mesh, bail out
                if obj.type != 'MESH':
                    obj = None
            else:
                # neither MESH or ARMATURE, bail out
                obj = None

    # Validate selected mesh, report errors if not okay
    if obj != None:
        mesh = obj.to_mesh()
        if mesh.uv_layers.active == None:
            operator.report({'ERROR'}, 'No UV layer found on model')
            obj.to_mesh_clear()
            return None

        obj.to_mesh_clear()
    else:
        operator.report({'ERROR'}, 'No model selected')
        return None

    return obj

def export_3df(context, filepath, obj, mat):
    depsgraph = context.evaluated_depsgraph_get()

    # get selected object and create mesh
    obj_matrix = obj.matrix_world
    obj_for_convert = obj.evaluated_get(depsgraph)
    mesh = obj_for_convert.to_mesh()

    # Transform mesh
    mesh.transform(mat @ obj_matrix)

    # UV layer
    uv_layer = mesh.uv_layers.active.data

    face_count = len(mesh.polygons)

    f = open(filepath, 'wb')

    # Write Header
    f.write(struct.pack("<LLLL", len(mesh.vertices), face_count, 1, 0)) # bones, texturesize

    # Write faces
    for poly in mesh.polygons:
        # Start of poly loop
        start = poly.loop_start
        count = poly.loop_total

        # Do our own triangulation; doing it using Blender causes a disconnect between
        # rig and mesh (deforming the object when playing the animation afterwards)
        v1 = start + 0
        for j in range(1, count -1):
            v2 = start + j
            v3 = start + j + 1

            f.write(
                struct.pack("<LLLLLLLLLHHLLLxxxxxxxxxxxx",
                mesh.loops[v1].vertex_index,
                mesh.loops[v2].vertex_index,
                mesh.loops[v3].vertex_index,
                int(uv_layer[v1].uv.x * 256),
                int(uv_layer[v2].uv.x * 256),
                int(uv_layer[v3].uv.x * 256),
                int(uv_layer[v1].uv.y * 256),
                int(uv_layer[v2].uv.y * 256),
                int(uv_layer[v3].uv.y * 256),
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

    # Write default bone
    f.write(struct.pack("<32sfffhH", b'default', 0, 0, 0, -1, 0)) # x,y,z, parent, hidden

    f.close()

    return {'FINISHED'}

def export_vtl(context, filepath, obj, frame_step, mat):
    # Exit edit mode before exporting, so current object states are exported properly.
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    # Get copy of active mesh
    scene = context.scene
    orig_frame = scene.frame_current
    scene_frames = range(scene.frame_start, scene.frame_end + 1, frame_step)

    fps = scene.render.fps / frame_step

    f = open(filepath, 'wb')
    f.write(struct.pack("<LLL", 0, int(fps), len(scene_frames))) # VCount, aniKPS, framesCount

    vcount = 0

    depsgraph = context.evaluated_depsgraph_get()

    obj_matrix = obj.matrix_world

    for frame in scene_frames:
        scene.frame_set(frame, subframe=0.0)

        # get selected object and create mesh
        obj_for_convert = obj.evaluated_get(depsgraph)
        mesh = obj_for_convert.to_mesh()

        # transform
        mesh.transform(mat @ obj_matrix)

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

    # restore frame to before export
    scene.frame_set(orig_frame, subframe=0.0)

    return {'FINISHED'}

# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper, orientation_helper, axis_conversion
from bpy.props import StringProperty, FloatProperty
from bpy.types import Operator
from mathutils import Matrix

@orientation_helper(axis_forward='-Z', axis_up='Y')
class ExportVTL(Operator, ExportHelper):
    """Export classic Carnivores VTL animation format"""
    bl_idname = "export_vtl.some_data"
    bl_label = "Export VTL (Carnivores)"
    bl_options = {'PRESET'}

    # ExportHelper mixin class uses this
    filename_ext = ".vtl"

    filter_glob: StringProperty(
        default="*.vtl",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.

    global_scale: FloatProperty(
        name="Scale",
        min=0.01, max=1000.0,
        default=1.0,
    )

    def execute(self, context):
        obj = prepare_export(context, self)
        if obj == None:
            return {'CANCELLED'}

        global_matrix = (
            Matrix.Scale(self.global_scale, 4) @
            axis_conversion(
                to_forward=self.axis_forward,
                to_up=self.axis_up,
            ).to_4x4()
        )

        return export_vtl(context, self.filepath, obj, 1, global_matrix)


@orientation_helper(axis_forward='-Z', axis_up='Y')
class Export3DF(Operator, ExportHelper):
    """Export classic Carnivores 3DF model format"""
    bl_idname = "export_3df.some_data"
    bl_label = "Export 3DF (Carnivores)"
    bl_options = {'PRESET'}

    # ExportHelper mixin class uses this
    filename_ext = ".3df"

    filter_glob: StringProperty(
        default="*.3df",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # List of operator properties, the attributes will be assigned
    # to the class instance from the operator settings before calling.
    global_scale: FloatProperty(
        name="Scale",
        min=0.01, max=1000.0,
        default=1.0,
    )

    def execute(self, context):
        obj = prepare_export(context, self)
        if obj == None:
            return {'CANCELLED'}

        global_matrix = (
            Matrix.Scale(self.global_scale, 4) @
            axis_conversion(
                to_forward=self.axis_forward,
                to_up=self.axis_up,
            ).to_4x4()
        )

        return export_3df(context, self.filepath, obj, global_matrix)

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
