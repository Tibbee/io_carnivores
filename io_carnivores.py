import os
import bpy
import struct

bl_info = {
    "name": "Carnivores Model/Animation Import/Export (3DF,VTL)",
    "author": "Ithamar R. Adema",
    "version": (1, 1, 0),
    "blender": (2, 82, 0),
    "description": "Import/Export plugin for classic Carnivores formats 3DF (base model), VTL (animations)",
    "category": "Import-Export",
}

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

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

    f = open(filepath, 'wb')

    # Write Header
    f.write(struct.pack("<LLLL", len(mesh.vertices), len(mesh.polygons), 1, 0)) # bones, texturesize

    # Write faces
    face_count = 0
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
            face_count += 1

    # Write Vertices
    for v in mesh.vertices:
        f.write(struct.pack("<fffHH", v.co.x, v.co.y, v.co.z, 0, 0)) # bone and hide

    # Write default bone
    f.write(struct.pack("<32sfffhH", b'default', 0, 0, 0, -1, 0)) # x,y,z, parent, hidden

    # Optionally fix up face count in header, due to quad/n-gon conversion
    if face_count != len(mesh.polygons):
        f.seek(4)
        f.write(struct.pack("<L", face_count))

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
from mathutils import Matrix, Vector

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

# -----------------------------------------------------------------------------
# Import
# -----------------------------------------------------------------------------

def bytes_to_string(bytes):
    return str(bytes.split(b'\0')[0], encoding='UTF-8')

def import_model(context, filepath, mat):
    f = open(filepath, 'rb')
    data = f.read()
    f.close()

    fname = os.path.basename(filepath)
    name, ext = os.path.splitext(fname)

    if ext.lower() == '.3df':
        return import_3df(data, name, mat)
    else:
        return import_car(data, mat)

def read_faces_and_verts(face_count, vert_count, data, offset):
    faces = []
    for i in range(face_count):
        face = struct.unpack_from('<IIIIIIIIIHHIII', data, offset)
        faces.append(face)
        offset += 64

    verts = []
    for i in range(vert_count):
        vert = struct.unpack_from('<fffhh', data, offset)
        verts.append(vert)
        offset += 16

    return offset, faces, verts

def read_texture(texture_size, data, offset):
    height = int((texture_size / 2) / 256)
    if height == 0:
        height = 256

    image = None
    if texture_size > 0 and offset + texture_size <= len(data):
        image = bpy.data.images.new('src', 256, height)
        p = [0.0] * 256 * height * 4
        # convert BGR5551 to RGBA
        dest = 0
        for pixels in range(int(texture_size / 2)):
            w = data[offset+0] + (data[offset+1] * 256)
            offset += 2
            b = (w >> 0) & 31
            g = (w >> 5) & 31
            r = (w >> 10) & 31
            p[dest + 0] = r / 31.0
            p[dest + 1] = g / 31.0
            p[dest + 2] = b / 31.0
            p[dest + 3] = 1.0
            dest += 4
        image.pixels[:] = p[:]
        image.pack()
        image.use_fake_user = True

    return height, image

def import_car(data, matrix):
    name_, anim_count, sfx_count, vert_count, face_count, texture_size = struct.unpack_from('<32sIIIII', data, 0)
    name = bytes_to_string(name_)
    offset, faces, verts = read_faces_and_verts(face_count, vert_count, data, 52)

    # Load texture
    height, image = read_texture(texture_size, data, offset)
    offset += texture_size
    
    vertgroups = [[] for s in range(256)]

    mverts = []
    for vidx in range(len(verts)):
        v = verts[vidx]
        mverts.append([ v[0], v[1], v[2] ])
        bone = v[3]
        if bone >= 0:
            vertgroups[bone].append(vidx)

    mfaces = []
    muvs = []
    for f in faces:
        mfaces.append([ f[0], f[1], f[2] ])
        muvs.append([ f[3] / 255, f[6] / (height-1) ])
        muvs.append([ f[4] / 255, f[7] / (height-1) ])
        muvs.append([ f[5] / 255, f[8] / (height-1) ])

    mesh_data = bpy.data.meshes.new(name + '_Mesh')
    mesh_data.from_pydata(mverts, [], mfaces)
    mesh_data.update()
    mesh_data.transform(matrix)

    obj = bpy.data.objects.new(name, mesh_data)

    # define vertex groups
    for vgi in range(len(vertgroups)):
        vg = vertgroups[vgi]
        if len(vg) > 0:
            grp = obj.vertex_groups.new(name="bone_%d" % vgi)
            grp.add(vg, 1.0, 'ADD')

    # Set UV mapping
    uv = obj.data.uv_layers.new(name=name + "_UV")
    for loop in obj.data.loops:
        uv.data[loop.index].uv = muvs[loop.index]

    # If we have a texture, setup material
    if image != None:
        mat = bpy.data.materials.new(name=name + "_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = image
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        obj.data.materials.append(mat)

    # Parse animations
    if anim_count > 0:     
        actions = []
        for i in range(anim_count):
            # parse anim header
            anim_name, kps, frame_count = struct.unpack_from('<32sII', data, offset)
            offset += 40
            anim_name = str(anim_name.split(b'\0')[0], encoding='UTF-8')

            action = bpy.data.actions.new(anim_name)
            fcurve = action.fcurves.new('eval_time')

            actions.append(action)

            for f in range(frame_count):
                sk = obj.shape_key_add(name=anim_name, from_mix=False)
                sk.interpolation = 'KEY_LINEAR'
                
                for j in range(vert_count):
                    x, y, z = struct.unpack_from('<hhh', data, offset)
                    offset += 6
                    sk.data[j].co.x = x / 16.0
                    sk.data[j].co.y = y / 16.0
                    sk.data[j].co.z = z / 16.0
                    sk.data[j].co = matrix @ sk.data[j].co

                fcurve.keyframe_points.insert(f, sk.frame)
                    
        obj.data.shape_keys.use_relative = False
        obj.data.shape_keys.animation_data_create()
        obj.data.shape_keys.animation_data.action = actions[0]
        
        anim_data = obj.data.shape_keys.animation_data
        for i in range(anim_count):
            trk = anim_data.nla_tracks.new()
            trk.name = actions[i].name
            strip = trk.strips.new(actions[i].name, int(actions[i].frame_range[0]), actions[i])
            # update start/end frames of action
            strip.action_frame_start = actions[i].frame_range[0]
            strip.action_frame_end = actions[i].frame_range[1]

    view_layer = bpy.context.view_layer
    collection = view_layer.active_layer_collection.collection
    collection.objects.link(obj)

    return {'FINISHED'}


def import_3df(data, name, mat):
    vert_count, face_count, bone_count, texture_size = struct.unpack_from('<IIII', data, 0)
    offset, faces, verts = read_faces_and_verts(face_count, vert_count, data, 16)

    # Parse model data
    bones = []
    for i in range(bone_count):
        bone = struct.unpack_from('<32sfffhH', data, offset)
        bones.append(bone)
        offset += 48
    
    # Load texture
    height, image = read_texture(texture_size, data, offset)
    
    vertgroups = [[] for s in range(256)]

    mverts = []
    for vidx in range(len(verts)):
        v = verts[vidx]
        mverts.append([ v[0], v[1], v[2] ])
        bone = v[3]
        if bone >= 0:
            vertgroups[bone].append(vidx)

    mfaces = []
    muvs = []
    for f in faces:
        mfaces.append([ f[0], f[1], f[2] ])
        muvs.append([ f[3] / 255, f[6] / (height-1) ])
        muvs.append([ f[4] / 255, f[7] / (height-1) ])
        muvs.append([ f[5] / 255, f[8] / (height-1) ])

    mesh_data = bpy.data.meshes.new(name + '_Mesh')
    mesh_data.from_pydata(mverts, [], mfaces)
    mesh_data.update()
    mesh_data.transform(mat)

    obj = bpy.data.objects.new(name, mesh_data)

    # If we have bones...
    if len(bones) > 0:
        # Create armature
        arm_data = bpy.data.armatures.new(name + "_arm_data")
        armobj = bpy.data.objects.new(name + "_Armature", arm_data)
        view_layer = bpy.context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.link(armobj)
        obj.parent = armobj
        # add armature modifier to mesh
        modifier = obj.modifiers.new(name + "Modifier", 'ARMATURE')
        modifier.object = armobj

        # create bones
        bpy.context.view_layer.objects.active = armobj
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        edit_bones = arm_data.edit_bones
        blbones = []
        for bi in range(len(bones)):
            bone = bones[bi]
            bone_name = bytes_to_string(bone[0])
            pos = mat @ Vector((bone[1], bone[2], bone[3]))
            b = edit_bones.new(bone_name)
            b.head = (pos.x, pos.y, pos.z)
            b.tail = (pos.x, pos.y, pos.z + 10)
            b.use_deform = True
            blbones.append(b)
            parent = bone[4]
            if parent != -1:
                b.tail = blbones[parent].head
                b.parent = blbones[parent]
        bpy.ops.object.mode_set(mode='OBJECT')

        # define vertex groups
        for vgi in range(len(bones)):
            vg = vertgroups[vgi]
            if len(vg) > 0:
                grp = obj.vertex_groups.new(name=bytes_to_string(bones[vgi][0]))
                grp.add(vg, 1.0, 'ADD')

    # Set UV mapping
    uv = obj.data.uv_layers.new(name=name+"_UV")
    for loop in obj.data.loops:
        uv.data[loop.index].uv = muvs[loop.index]

    # If we have a texture, setup material
    if image != None:
        mat = bpy.data.materials.new(name=name + "_Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = image
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        obj.data.materials.append(mat)

    view_layer = bpy.context.view_layer
    collection = view_layer.active_layer_collection.collection
    collection.objects.link(obj)

    return {'FINISHED'}


# ImportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ImportHelper, orientation_helper, axis_conversion
from bpy.props import StringProperty, FloatProperty
from bpy.types import Operator
from mathutils import Matrix, Vector

@orientation_helper(axis_forward='Z', axis_up='Y')
class Import3DF(Operator, ImportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "import3df.some_data"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Carnivores (3DF)"

    # ImportHelper mixin class uses this
    filename_ext = ".3df;.car"

    filter_glob: StringProperty(
        default="*.3df;*.car",
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
        global_matrix = (
            Matrix.Scale(self.global_scale, 4) @
            axis_conversion(
                to_forward=self.axis_forward,
                to_up=self.axis_up,
            ).to_4x4()
        )

        return import_model(context, self.filepath, global_matrix)

# Only needed if you want to add into a dynamic menu.
def menu_func_import(self, context):
    self.layout.operator(Import3DF.bl_idname, text="3DF/CAR (Carnivores)")

def register():
    bpy.utils.register_class(Export3DF)
    bpy.utils.register_class(ExportVTL)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.utils.register_class(Import3DF)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(Export3DF)
    bpy.utils.unregister_class(ExportVTL)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.utils.unregister_class(Import3DF)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
