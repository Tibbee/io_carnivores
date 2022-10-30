# Development Notes

I'm storing links and code snippets here related to Blender scripting, to save me googling in the future ;)

# TODO

## Export multiple objects

Allow multiple meshes to be exported, and create a dummy bone per mesh. This is so every "sub mesh" can get its own collision in-game.

## 3DF import

Create mesh, convert face-uv to vertex-uv. Look into creating Armature and weights based on the 3DF.

## CAR import

Use 3DF import code for base mesh, import animation frames as shape keys.

# Links

## Animation

- Keyframe information: https://blender.stackexchange.com/questions/27889/how-to-find-number-of-animated-frames-in-a-scene-via-python
- Multiple animation export: https://blender.stackexchange.com/questions/3281/how-do-i-export-animation-data-stored-in-actions-using-the-python-api

## Shape keys

- Shape key creation: https://blender.stackexchange.com/questions/111661/creating-shape-keys-using-python

## Armature/Bones

- https://blender.stackexchange.com/questions/51684/python-create-custom-armature-without-ops
- https://devtalk.blender.org/t/vertex-weights-and-indices-for-a-bmesh/1457
- https://devtalk.blender.org/t/help-implementing-animation-in-blender-2-8-import-addon/7093
