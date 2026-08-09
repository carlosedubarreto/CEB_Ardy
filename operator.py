import bpy
import os
import sys
import subprocess
import math
import mathutils
import gpu
import blf
import socket
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent

try:
    import numpy as np
except ImportError:
    pass

_realtime_client = None
_realtime_running = False
_active_stream_operator = None
_overlay_draw_handler = None
_3d_draw_handler = None

def tag_redraw_view3d(self=None, context=None):
    if context is None:
        if isinstance(self, bpy.types.Context):
            context = self
        else:
            context = bpy.context
    if hasattr(context, "window_manager") and context.window_manager:
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

def send_waypoints_to_bridge(context=None, start_frame=None):
    global _realtime_client, _realtime_running, _active_stream_operator
    if not _realtime_client or not _realtime_running:
        return
    if context is None:
        context = bpy.context
    char = get_active_character(context)
    if not char:
        return
    
    if start_frame is None:
        if _active_stream_operator is not None and hasattr(_active_stream_operator, "_start_frame"):
            start_frame = _active_stream_operator._start_frame
        else:
            start_frame = context.scene.frame_current

    try:
        _realtime_client.sendall(b"CLEAR_WAYPOINTS\n")
        for item in char.prompt_schedule:
            if item.enabled and getattr(item, "has_waypoint", False):
                if item.start_frame < start_frame:
                    continue
                if item.waypoint_object_name:
                    wp_obj = bpy.data.objects.get(item.waypoint_object_name)
                    if wp_obj:
                        global_co = wp_obj.matrix_world.to_translation()
                    else:
                        global_co = mathutils.Vector(item.waypoint_co)
                else:
                    global_co = mathutils.Vector(item.waypoint_co)

                cmd = f"WAYPOINT:{item.start_frame}:{global_co[0]:.4f}:{global_co[1]:.4f}:{global_co[2]:.4f}\n"
                _realtime_client.sendall(cmd.encode("utf-8"))
    except Exception as e:
        print(f"[CEB Ardy] Error sending waypoints over socket: {e}")

def blender_pos_to_ardy(pos_b, scale=1.0):
    """Convert Blender position (X-Right, Y-Forward, Z-Up) to ARDY position (X-Left, Y-Up, Z-Forward)."""
    return [-float(pos_b[0]) / scale, float(pos_b[2]) / scale, float(pos_b[1]) / scale]

def blender_rot_to_ardy(R_b):
    """Convert Blender 3x3 rotation matrix to ARDY 3x3 rotation matrix using M = [[-1,0,0],[0,0,1],[0,1,0]]."""
    return [
        [ float(R_b[0][0]), -float(R_b[0][2]), -float(R_b[0][1])],
        [-float(R_b[2][0]),  float(R_b[2][2]),  float(R_b[2][1])],
        [-float(R_b[1][0]),  float(R_b[1][2]),  float(R_b[1][1])],
    ]


# ARDY joint name orderings (must appear before send_pose_constraints_to_bridge)
_soma30_names = [
    'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw', 'LeftEye', 'RightEye',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumbEnd', 'LeftHandMiddleEnd',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd',
    'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase'
]
_smpl24_names = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
    'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]
_smpl22_names = _smpl24_names[:22]

def get_char_prefix(char):
    if not char:
        return "C1"
    name = char.name.strip()
    import re
    match = re.match(r'^(?:Character|Char)[_\s-]?(\d+)$', name, re.IGNORECASE)
    if match:
        return f"C{match.group(1)}"
    
    words = name.replace("_", " ").replace("-", " ").split()
    if len(words) > 1:
        prefix = "".join(w[0].upper() for w in words if w)
        if words[-1].isdigit():
            prefix = prefix[:-1] + words[-1]
        return prefix
    else:
        match = re.search(r'(\d+)$', name)
        if match:
            num = match.group(1)
            non_num = name[:-len(num)]
            return (non_num[0].upper() if non_num else "") + num
        return name[:3].upper()

def remove_pose_constraint_armature(item):
    obj_name = getattr(item, "pose_armature_name", "")
    if obj_name:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception as e:
                print(f"[CEB Ardy] Error removing pose constraint armature: {e}")

    # Fallback search matching ARDY_Pose_F<frame> or <prefix>_Ardy_Pose_F<frame>
    for obj in list(bpy.data.objects):
        if obj.type == 'ARMATURE':
            is_match = False
            if obj.name.startswith(f"ARDY_Pose_F{item.start_frame}"):
                is_match = True
            elif "Ardy_Pose_F" in obj.name:
                parts = obj.name.split("Ardy_Pose_F")
                if len(parts) > 1:
                    frame_part = parts[1].split("_")[0]
                    if frame_part.isdigit() and int(frame_part) == item.start_frame:
                        is_match = True
            if is_match:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

    item.pose_armature_name = ""

def get_or_create_pose_constraint_armature(context, item, copy_current_pose=True):
    obj_name = getattr(item, "pose_armature_name", "")
    obj = bpy.data.objects.get(obj_name) if obj_name else None

    if not obj:
        char_arm = None
        char = get_active_character(context)
        if char and char.arm_obj_name:
            candidate_obj = bpy.data.objects.get(char.arm_obj_name)
            if candidate_obj:
                char_arm = candidate_obj

        if not char_arm and context.active_object and context.active_object.type == 'ARMATURE' and not context.active_object.name.startswith("ARDY_Pose_") and not "_Ardy_Pose_F" in context.active_object.name:
            char_arm = context.active_object
        
        if not char_arm:
            for candidate in ["Core_Armature", "SOMA_Armature"]:
                candidate_obj = bpy.data.objects.get(candidate)
                if candidate_obj:
                    char_arm = candidate_obj
                    break
        if not char_arm:
            for o in context.scene.objects:
                if o.type == 'ARMATURE' and not o.name.startswith("ARDY_Pose_") and not "_Ardy_Pose_F" in o.name:
                    char_arm = o
                    break

        if not char_arm:
            print("[CEB Ardy] No character armature found to duplicate for pose constraint.")
            return None

        prefix = get_char_prefix(char)
        base_name = f"{prefix}_Ardy_Pose_F{item.start_frame}"
        obj_name = base_name
        idx = 1
        while bpy.data.objects.get(obj_name):
            obj_name = f"{base_name}_{idx}"
            idx += 1

        arm_data_copy = char_arm.data.copy()
        arm_data_copy.name = f"{obj_name}_Data"
        arm_data_copy.display_type = 'STICK'
        obj = bpy.data.objects.new(obj_name, arm_data_copy)
        obj.matrix_world = char_arm.matrix_world.copy()
        obj.show_in_front = True
        obj.display_type = 'WIRE'

        col_name = "ARDY_PoseConstraints"
        collection = bpy.data.collections.get(col_name)
        if not collection:
            collection = bpy.data.collections.new(col_name)
            context.scene.collection.children.link(collection)
        collection.objects.link(obj)
        if hasattr(context, "view_layer") and context.view_layer:
            context.view_layer.update()

        if copy_current_pose and char_arm.pose and obj.pose:
            for src_bone in char_arm.pose.bones:
                tgt_bone = obj.pose.bones.get(src_bone.name)
                if tgt_bone:
                    tgt_bone.rotation_mode = src_bone.rotation_mode
                    tgt_bone.location = src_bone.location.copy()
                    tgt_bone.rotation_quaternion = src_bone.rotation_quaternion.copy()
                    tgt_bone.rotation_euler = src_bone.rotation_euler.copy()
                    tgt_bone.rotation_axis_angle = list(src_bone.rotation_axis_angle)
                    tgt_bone.scale = src_bone.scale.copy()

        item.pose_armature_name = obj.name

        char = get_active_character(context)
        if char:
            crowd = get_crowd_for_character(char.name, context)
            if crowd and crowd.empty_object_name:
                crowd_empty = bpy.data.objects.get(crowd.empty_object_name)
                if crowd_empty:
                    obj.parent = crowd_empty
                    obj.matrix_parent_inverse = crowd_empty.matrix_world.inverted()

    return obj

def send_pose_constraints_to_bridge(context=None, start_frame=None):
    global _realtime_client, _realtime_running, _active_stream_operator
    if not _realtime_client or not _realtime_running:
        return
    if context is None:
        context = bpy.context
    if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
        return
    props = context.scene.ceb_ardy
    char = get_active_character(context)
    if not char:
        return
    
    if start_frame is None:
        if _active_stream_operator is not None and hasattr(_active_stream_operator, "_start_frame"):
            start_frame = _active_stream_operator._start_frame
        else:
            start_frame = context.scene.frame_current

    scale = props.import_scale
    try:
        _realtime_client.sendall(b"CLEAR_POSE_CONSTRAINTS\n")
        import json

        # 1. Send current viewport pose as starting constraint at start_frame
        has_existing_start_constraint = any(
            item.enabled and getattr(item, "has_pose_constraint", False) and item.start_frame == start_frame
            for item in char.prompt_schedule
        )

        if not has_existing_start_constraint:
            arm_obj = None
            if char.arm_obj_name:
                arm_obj = bpy.data.objects.get(char.arm_obj_name)
            if not arm_obj:
                clean_name = char.name.replace(" ", "_")
                arm_obj = bpy.data.objects.get(f"{clean_name}_Armature")

            if arm_obj and arm_obj.pose:
                bone_names = [b.name for b in arm_obj.pose.bones]
                num_bones = len(bone_names)
                if num_bones == 30:
                    joint_order = _soma30_names
                elif num_bones == 24:
                    joint_order = _smpl24_names
                elif num_bones == 22:
                    joint_order = _smpl22_names
                else:
                    joint_order = bone_names

                joints_pos = []
                joints_rot = []
                for jname in joint_order:
                    bone = arm_obj.pose.bones.get(jname)
                    if bone is None:
                        joints_pos.append([0.0, 0.0, 0.0])
                        joints_rot.append([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                        continue

                    # Capture absolute world space transform for ARDY constraints
                    mat_world = arm_obj.matrix_world @ bone.matrix
                    head_loc = mat_world.to_translation()
                    rot_mat = mat_world.to_3x3()

                    pos_a = blender_pos_to_ardy(head_loc, scale=scale)
                    rot_a = blender_rot_to_ardy(rot_mat)

                    joints_pos.append(pos_a)
                    joints_rot.append(rot_a)

                pos_json = json.dumps(joints_pos)
                rot_json = json.dumps(joints_rot)
                cmd = f"POSE_CONSTRAINT:{start_frame}:{pos_json}:{rot_json}\n"
                _realtime_client.sendall(cmd.encode("utf-8"))
                print(f"[CEB Ardy] Sent initial/current viewport pose constraint for frame {start_frame}")

        # 2. Send prompt_schedule pose constraints
        for item in char.prompt_schedule:
            if item.enabled and getattr(item, "has_pose_constraint", False) and getattr(item, "pose_armature_name", ""):
                if item.start_frame < start_frame:
                    continue
                arm_obj = bpy.data.objects.get(item.pose_armature_name)
                if not arm_obj:
                    continue

                bone_names = [b.name for b in arm_obj.pose.bones]
                num_bones = len(bone_names)

                if num_bones == 30:
                    joint_order = _soma30_names
                elif num_bones == 24:
                    joint_order = _smpl24_names
                elif num_bones == 22:
                    joint_order = _smpl22_names
                else:
                    joint_order = bone_names

                joints_pos = []
                joints_rot = []
                for jname in joint_order:
                    bone = arm_obj.pose.bones.get(jname)
                    if bone is None:
                        joints_pos.append([0.0, 0.0, 0.0])
                        joints_rot.append([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                        continue

                    # Capture absolute world space transform for ARDY constraints
                    mat_world = arm_obj.matrix_world @ bone.matrix
                    head_loc = mat_world.to_translation()
                    rot_mat = mat_world.to_3x3()

                    pos_a = blender_pos_to_ardy(head_loc, scale=scale)
                    rot_a = blender_rot_to_ardy(rot_mat)

                    joints_pos.append(pos_a)
                    joints_rot.append(rot_a)

                pos_json = json.dumps(joints_pos)
                rot_json = json.dumps(joints_rot)
                cmd = f"POSE_CONSTRAINT:{item.start_frame}:{pos_json}:{rot_json}\n"
                _realtime_client.sendall(cmd.encode("utf-8"))
                print(f"[CEB Ardy] Sent scheduled pose constraint for frame {item.start_frame} ({num_bones} joints)")
    except Exception as e:
        print(f"[CEB Ardy] Error sending pose constraints over socket: {e}")

def update_has_pose_constraint(self, context):
    tag_redraw_view3d(context)
    if getattr(self, "has_pose_constraint", False):
        get_or_create_pose_constraint_armature(context, self)
    else:
        remove_pose_constraint_armature(self)
    send_pose_constraints_to_bridge(context)

def remove_waypoint_empty(item):
    obj_name = getattr(item, "waypoint_object_name", "")
    if obj_name:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception as e:
                print(f"[CEB Ardy] Error removing waypoint empty: {e}")

    # Fallback search matching ARDY_Waypoint_F<frame> or <prefix>_Ardy_waypoint_F<frame>
    for obj in list(bpy.data.objects):
        if obj.type == 'EMPTY':
            is_match = False
            if obj.name.startswith(f"ARDY_Waypoint_F{item.start_frame}"):
                is_match = True
            elif "Ardy_waypoint_F" in obj.name:
                parts = obj.name.split("Ardy_waypoint_F")
                if len(parts) > 1:
                    frame_part = parts[1].split("_")[0]
                    if frame_part.isdigit() and int(frame_part) == item.start_frame:
                        is_match = True
            if is_match:
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

    item.waypoint_object_name = ""

def update_has_waypoint(self, context):
    tag_redraw_view3d(context)
    if getattr(self, "has_waypoint", False):
        if abs(self.waypoint_co[0]) < 1e-5 and abs(self.waypoint_co[1]) < 1e-5 and abs(self.waypoint_co[2]) < 1e-5:
            char = get_active_character(context)
            if char:
                cx, cy, cz, ch = get_character_world_transform(char)
                fwd_x = -math.sin(ch)
                fwd_y = math.cos(ch)
                self.waypoint_co = (cx + fwd_x * 2.0, cy + fwd_y * 2.0, cz)
        get_or_create_waypoint_empty(context, self)
    else:
        remove_waypoint_empty(self)
    send_waypoints_to_bridge(context)

def sync_prompt_item_object_names(item, context=None):
    if context is None:
        context = bpy.context
    char = get_active_character(context)
    prefix = get_char_prefix(char)

    crowd = get_crowd_for_character(char.name, context) if char else None
    crowd_empty = bpy.data.objects.get(crowd.empty_object_name) if (crowd and crowd.empty_object_name) else None
    inv_mat = crowd_empty.matrix_world.inverted() if crowd_empty else None

    # 1. Update Pose Constraint Armature Object Name
    if getattr(item, "has_pose_constraint", False):
        arm_name = getattr(item, "pose_armature_name", "")
        obj = bpy.data.objects.get(arm_name) if arm_name else None
        
        # Fallback search specifically matching this item's frame if exact name lookup missed
        if not obj:
            target_suffix = f"Ardy_Pose_F{item.start_frame}"
            for candidate in list(bpy.data.objects):
                if candidate.type == 'ARMATURE' and target_suffix in candidate.name:
                    obj = candidate
                    break

        if obj:
            target_base_name = f"{prefix}_Ardy_Pose_F{item.start_frame}"
            if obj.name != target_base_name:
                new_name = target_base_name
                idx = 1
                while bpy.data.objects.get(new_name) and bpy.data.objects.get(new_name) != obj:
                    new_name = f"{target_base_name}_{idx}"
                    idx += 1
                
                obj.name = new_name
                if obj.data:
                    obj.data.name = f"{new_name}_Data"
                item.pose_armature_name = obj.name
                print(f"[CEB Ardy] Automatically updated pose constraint armature name to '{obj.name}'")

            if crowd_empty and obj.parent != crowd_empty:
                obj.parent = crowd_empty
                obj.matrix_parent_inverse = inv_mat

    # 2. Update Waypoint Empty Object Name
    if getattr(item, "has_waypoint", False):
        wp_name = getattr(item, "waypoint_object_name", "")
        wp_obj = bpy.data.objects.get(wp_name) if wp_name else None

        # Fallback search specifically matching this item's frame if exact name lookup missed
        if not wp_obj:
            target_suffix = f"Ardy_waypoint_F{item.start_frame}"
            for candidate in list(bpy.data.objects):
                if candidate.type == 'EMPTY' and target_suffix in candidate.name:
                    wp_obj = candidate
                    break

        if wp_obj:
            target_base_name = f"{prefix}_Ardy_waypoint_F{item.start_frame}"
            if wp_obj.name != target_base_name:
                new_name = target_base_name
                idx = 1
                while bpy.data.objects.get(new_name) and bpy.data.objects.get(new_name) != wp_obj:
                    new_name = f"{target_base_name}_{idx}"
                    idx += 1

                wp_obj.name = new_name
                item.waypoint_object_name = wp_obj.name
                print(f"[CEB Ardy] Automatically updated waypoint object name to '{wp_obj.name}'")

            if crowd_empty and wp_obj.parent != crowd_empty:
                wp_obj.parent = crowd_empty
                wp_obj.matrix_parent_inverse = inv_mat

def update_prompt_item(self, context):
    global _realtime_client, _active_stream_operator
    sync_prompt_item_object_names(self, context)
    tag_redraw_view3d(context)
    send_waypoints_to_bridge(context)
    send_pose_constraints_to_bridge(context)
    if _realtime_client and context and hasattr(context, "scene") and hasattr(context.scene, "ceb_ardy"):
        char = get_active_character(context)
        if char:
            current_frame = context.scene.frame_current
            active_prompt = get_active_prompt_for_frame(context.scene.ceb_ardy, current_frame)
            try:
                prompt_cmd = f"PROMPT:{active_prompt}\n"
                _realtime_client.sendall(prompt_cmd.encode("utf-8"))
                if _active_stream_operator is not None and hasattr(_active_stream_operator, "_frame_queue"):
                    if len(_active_stream_operator._frame_queue) > 3:
                        _active_stream_operator._frame_queue = _active_stream_operator._frame_queue[:3]
                    _active_stream_operator._last_sent_prompt = active_prompt
                print(f"[CEB Ardy] Prompt item updated & sent → '{active_prompt}'")
            except Exception as e:
                print(f"[CEB Ardy] Failed to send updated prompt: {e}")

def update_realtime_prompt(self, context):
    global _realtime_client, _active_stream_operator
    if _realtime_client:
        try:
            prompt_cmd = f"PROMPT:{self.realtime_prompt}\n"
            _realtime_client.sendall(prompt_cmd.encode("utf-8"))
            if _active_stream_operator is not None and hasattr(_active_stream_operator, "_frame_queue"):
                if len(_active_stream_operator._frame_queue) > 3:
                    _active_stream_operator._frame_queue = _active_stream_operator._frame_queue[:3]
                _active_stream_operator._last_sent_prompt = self.realtime_prompt
            print(f"[CEB Ardy] Prompt update sent → '{self.realtime_prompt}'")
        except Exception as e:
            print(f"[CEB Ardy] Failed to send prompt over socket: {e}")

def update_overlay_visibility(self, context):
    tag_redraw_view3d(context)

def update_waypoint_co(self, context):
    tag_redraw_view3d(context)
    if getattr(self, "has_waypoint", False) and getattr(self, "waypoint_object_name", ""):
        obj = bpy.data.objects.get(self.waypoint_object_name)
        if obj:
            target_co = mathutils.Vector(self.waypoint_co)
            if obj.parent:
                obj.matrix_world.translation = target_co
            elif (obj.location - target_co).length > 1e-4:
                obj.location = target_co
    send_waypoints_to_bridge(context)

class CEB_Ardy_PromptItem(bpy.types.PropertyGroup):
    prompt: bpy.props.StringProperty(
        name="Prompt",
        description="Prompt text for motion generation",
        default="walk",
        update=update_prompt_item
    )
    start_frame: bpy.props.IntProperty(
        name="Start Frame",
        description="Frame number when this prompt starts running",
        default=0,
        min=0,
        update=update_prompt_item
    )
    enabled: bpy.props.BoolProperty(
        name="Enabled",
        description="Whether this prompt entry is enabled",
        default=True,
        update=update_prompt_item
    )
    has_waypoint: bpy.props.BoolProperty(
        name="Has Waypoint",
        description="Whether this prompt entry has a 3D Waypoint location target attached",
        default=False,
        update=update_has_waypoint
    )
    waypoint_co: bpy.props.FloatVectorProperty(
        name="Waypoint Location",
        description="3D world location vector (X, Y, Z) for this waypoint",
        subtype='TRANSLATION',
        size=3,
        default=(0.0, 0.0, 0.0),
        update=update_waypoint_co
    )
    waypoint_object_name: bpy.props.StringProperty(
        name="Empty Object Name",
        description="Name of the Blender Empty object linked to this waypoint",
        default=""
    )
    has_pose_constraint: bpy.props.BoolProperty(
        name="Has Pose Constraint",
        description="Whether this entry has a full-body pose constraint attached",
        default=False,
        update=update_has_pose_constraint
    )
    pose_armature_name: bpy.props.StringProperty(
        name="Pose Armature Name",
        description="Name of the duplicated ghost armature object used for pose constraints",
        default=""
    )

class CEB_Ardy_Character(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        name="Name",
        description="Character name",
        default="Character_1",
        update=tag_redraw_view3d
    )
    model: bpy.props.EnumProperty(
        name="Model",
        description="Model to use for motion",
        items=[
            ('core', "CORE", "CORE 27-joint skeleton")
        ],
        default='core'
    )
    realtime_prompt: bpy.props.StringProperty(
        name="Live Prompt",
        description="Text prompt sent to ARDY in real-time",
        default="walk",
        update=update_realtime_prompt
    )
    prompt_schedule: bpy.props.CollectionProperty(
        type=CEB_Ardy_PromptItem
    )
    prompt_schedule_index: bpy.props.IntProperty(
        name="Active Prompt Index",
        default=0
    )
    arm_obj_name: bpy.props.StringProperty(default="")
    mesh_obj_name: bpy.props.StringProperty(default="")
    parent_obj_name: bpy.props.StringProperty(default="")

def update_crowd_hide(self, context):
    tag_redraw_view3d(context)
    hide_val = getattr(self, "hide_viewport", False)
    
    # 1. Hide/Unhide the Crowd Empty object
    if self.empty_object_name:
        empty_obj = bpy.data.objects.get(self.empty_object_name)
        if empty_obj:
            empty_obj.hide_viewport = hide_val
            empty_obj.hide_set(hide_val)

    # 2. Hide/Unhide all characters belonging to this crowd
    char_names = [n.strip() for n in self.character_names.split(",") if n.strip()]
    props = getattr(context.scene, "ceb_ardy", None) if hasattr(context, "scene") else None
    
    for c_name in char_names:
        clean_prefix = c_name.replace(" ", "_")
        char_item = None
        if props:
            for c in props.characters:
                if c.name == c_name:
                    char_item = c
                    break

        arm_name = char_item.arm_obj_name if (char_item and char_item.arm_obj_name) else f"{clean_prefix}_Armature"
        mesh_name = char_item.mesh_obj_name if (char_item and char_item.mesh_obj_name) else f"{clean_prefix}_Skin"

        for o_name in (arm_name, mesh_name):
            obj = bpy.data.objects.get(o_name)
            if obj:
                obj.hide_viewport = hide_val
                obj.hide_set(hide_val)

        if char_item:
            for pitem in char_item.prompt_schedule:
                if pitem.waypoint_object_name:
                    wp_obj = bpy.data.objects.get(pitem.waypoint_object_name)
                    if wp_obj:
                        wp_obj.hide_viewport = hide_val
                        wp_obj.hide_set(hide_val)
                if getattr(pitem, "pose_armature_name", ""):
                    pc_obj = bpy.data.objects.get(pitem.pose_armature_name)
                    if pc_obj:
                        pc_obj.hide_viewport = hide_val
                        pc_obj.hide_set(hide_val)

def get_crowd_for_character(char_name, context=None):
    if context is None:
        context = bpy.context
    if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
        return None
    props = context.scene.ceb_ardy
    for crowd in props.crowds:
        char_names = [n.strip() for n in crowd.character_names.split(",") if n.strip()]
        if char_name in char_names:
            return crowd
    return None

def parent_character_items_to_crowd(char, crowd_empty):
    if not char or not crowd_empty:
        return

    # Ensure view layer dependency graph is updated so crowd_empty.matrix_world is valid
    if hasattr(bpy.context, "view_layer") and bpy.context.view_layer:
        bpy.context.view_layer.update()
    else:
        crowd_empty.matrix_world = mathutils.Matrix.Translation(crowd_empty.location)

    inv_mat = crowd_empty.matrix_world.inverted()

    # Parent character armature
    clean_prefix = char.name.replace(" ", "_")
    arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_prefix}_Armature"
    arm_obj = bpy.data.objects.get(arm_name)
    if arm_obj and arm_obj.parent != crowd_empty:
        wmat = arm_obj.matrix_world.copy()
        arm_obj.parent = crowd_empty
        arm_obj.matrix_parent_inverse = inv_mat
        arm_obj.matrix_world = wmat

    # Parent waypoints & pose constraints belonging to this character
    for pitem in char.prompt_schedule:
        if pitem.waypoint_object_name:
            wp_obj = bpy.data.objects.get(pitem.waypoint_object_name)
            if wp_obj and wp_obj.parent != crowd_empty:
                wmat = wp_obj.matrix_world.copy()
                wp_obj.parent = crowd_empty
                wp_obj.matrix_parent_inverse = inv_mat
                wp_obj.matrix_world = wmat
        if getattr(pitem, "pose_armature_name", ""):
            pc_obj = bpy.data.objects.get(pitem.pose_armature_name)
            if pc_obj and pc_obj.parent != crowd_empty:
                wmat = pc_obj.matrix_world.copy()
                pc_obj.parent = crowd_empty
                pc_obj.matrix_parent_inverse = inv_mat
                pc_obj.matrix_world = wmat

def update_active_crowd(self, context):
    tag_redraw_view3d(context)
    if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
        return
    props = context.scene.ceb_ardy
    crowd = get_active_crowd(context)
    if crowd and crowd.empty_object_name:
        empty_obj = bpy.data.objects.get(crowd.empty_object_name)
        if empty_obj and not crowd.hide_viewport:
            try:
                if context.active_object and context.active_object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                for obj in context.view_layer.objects:
                    obj.select_set(False)
                empty_obj.select_set(True)
                context.view_layer.objects.active = empty_obj
            except Exception:
                pass

class CEB_Ardy_CrowdItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        name="Name",
        description="Crowd name",
        default="Crowd_1",
        update=tag_redraw_view3d
    )
    empty_object_name: bpy.props.StringProperty(
        name="Parent Empty Object",
        default=""
    )
    character_names: bpy.props.StringProperty(
        name="Character Names",
        description="Comma-separated names of characters in this crowd",
        default=""
    )
    hide_viewport: bpy.props.BoolProperty(
        name="Hide Crowd in Viewport",
        description="Toggle visibility of this crowd and all its characters in the 3D Viewport",
        default=False,
        update=update_crowd_hide
    )
    clear_settings_before_generate: bpy.props.BoolProperty(
        name="Clear Settings Before Generate",
        description="If enabled, removes all waypoints and pose constraints for the crowd's characters before generating. If disabled, re-runs streaming using existing positions, waypoints, and constraints without re-copying or re-calculating crowd layout.",
        default=True
    )
    crowd_count: bpy.props.IntProperty(default=4)
    layout_mode: bpy.props.StringProperty(default='GRID')
    spacing: bpy.props.FloatProperty(default=2.5)
    start_frame: bpy.props.IntProperty(default=1)
    end_frame: bpy.props.IntProperty(default=250)
    source_char_name: bpy.props.StringProperty(default="")

def get_active_character(context=None):
    if context is None:
        context = bpy.context
    if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
        return None
    props = context.scene.ceb_ardy
    if 0 <= props.active_character_index < len(props.characters):
        return props.characters[props.active_character_index]
    elif len(props.characters) > 0:
        return props.characters[0]
    return None

def get_active_crowd(context=None):
    if context is None:
        context = bpy.context
    if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
        return None
    props = context.scene.ceb_ardy
    if 0 <= props.active_crowd_index < len(props.crowds):
        return props.crowds[props.active_crowd_index]
    elif len(props.crowds) > 0:
        return props.crowds[0]
    return None

def get_active_prompt_for_frame(props=None, frame=0, context=None):
    char = get_active_character(context)
    if char and len(char.prompt_schedule) > 0:
        text_items = [item for item in char.prompt_schedule if item.enabled and item.prompt and item.prompt.strip() and not getattr(item, "has_waypoint", False)]
        if text_items:
            sorted_schedule = sorted(text_items, key=lambda x: x.start_frame)
            active_prompt = None
            for item in sorted_schedule:
                if frame >= item.start_frame:
                    active_prompt = item.prompt
            if active_prompt:
                return active_prompt
    if char and char.realtime_prompt:
        return char.realtime_prompt
    return "walk"

def get_character_world_transform(char):
    char_x, char_y, char_z = 0.0, 0.0, 0.0
    char_heading = 0.0
    
    if not char:
        return char_x, char_y, char_z, char_heading

    # Ensure view layer dependency graph is updated so matrix_world is accurate
    try:
        if bpy.context and hasattr(bpy.context, "view_layer") and bpy.context.view_layer:
            bpy.context.view_layer.update()
    except Exception:
        pass

    clean_name = char.name.replace(" ", "_")
    arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_name}_Armature"
    arm_obj = bpy.data.objects.get(arm_name)
    
    if arm_obj:
        world_matrix = arm_obj.matrix_world
        world_loc = world_matrix.to_translation()
        char_x = world_loc.x
        char_y = world_loc.y
        char_z = world_loc.z
        char_heading = world_matrix.to_euler().z
        # Fallback to direct location if matrix_world is zero but location is set
        if abs(char_x) < 1e-5 and abs(char_y) < 1e-5 and (abs(arm_obj.location.x) > 1e-4 or abs(arm_obj.location.y) > 1e-4):
            char_x = arm_obj.location.x
            char_y = arm_obj.location.y
            char_z = arm_obj.location.z
            char_heading = arm_obj.rotation_euler.z
    else:
        parent_name = char.parent_obj_name if char.parent_obj_name else f"ARDY_Character_{clean_name}"
        parent_obj = bpy.data.objects.get(parent_name)
        if parent_obj:
            world_loc = parent_obj.matrix_world.to_translation()
            char_x = world_loc.x
            char_y = world_loc.y
            char_z = world_loc.z
            char_heading = parent_obj.matrix_world.to_euler().z
            
    return char_x, char_y, char_z, char_heading


_reset_id_counter = 0

def send_switch_char_cmd(char, active_prompt, frame, context):
    global _reset_id_counter, _realtime_client, _active_stream_operator
    _reset_id_counter += 1
    reset_id = _reset_id_counter

    if _active_stream_operator is not None:
        _active_stream_operator._current_reset_id = reset_id
        _active_stream_operator._frame_queue = []
        _active_stream_operator._buffer = ""
        _active_stream_operator._reset_pending = True

    char_x, char_y, char_z, char_heading = get_character_world_transform(char)
    model = char.model if char else 'core'
    char_name = char.name if char else 'Character_1'
    cmd = f"SWITCH_CHAR:{char_name}:{model}:{active_prompt}:{frame}:{char_x:.4f}:{char_y:.4f}:{char_z:.4f}:{char_heading:.4f}:{reset_id}\n"
    
    if _realtime_client:
        try:
            _realtime_client.sendall(cmd.encode("utf-8"))
            if _active_stream_operator is not None:
                _active_stream_operator._start_frame = frame
                _active_stream_operator._last_sent_prompt = active_prompt
            send_waypoints_to_bridge(context, start_frame=frame)
            send_pose_constraints_to_bridge(context, start_frame=frame)
            print(f"[CEB Ardy] Sent SWITCH_CHAR for '{char_name}' at pos=({char_x:.2f},{char_y:.2f},{char_z:.2f}), reset_id={reset_id}")
        except Exception as e:
            print(f"[CEB Ardy] Error sending SWITCH_CHAR command: {e}")
    return reset_id

def update_active_character(self, context):
    tag_redraw_view3d(context)
    
    char = get_active_character(context)
    if char and hasattr(context, "scene") and hasattr(context.scene, "ceb_ardy"):
        crowd = get_crowd_for_character(char.name, context)
        if crowd and crowd.hide_viewport:
            try:
                if context.active_object and context.active_object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                for o in context.view_layer.objects:
                    o.select_set(False)
            except Exception:
                pass
            return

        clean_name = char.name.replace(" ", "_")
        arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_name}_Armature"
        arm_obj = bpy.data.objects.get(arm_name)
        if arm_obj:
            try:
                # Switch to object mode first if we are in another mode
                if context.active_object and context.active_object.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                
                # Deselect all objects
                for o in context.view_layer.objects:
                    o.select_set(False)
                    
                # Select the armature and make it active
                arm_obj.select_set(True)
                context.view_layer.objects.active = arm_obj
            except Exception as select_err:
                print(f"[CEB Ardy] Failed to select armature '{arm_name}' in viewport: {select_err}")

    global _realtime_client, _active_stream_operator
    if _realtime_client:
        if char:
            current_frame = context.scene.frame_current if hasattr(context, "scene") else 0
            active_prompt = get_active_prompt_for_frame(context.scene.ceb_ardy, current_frame, context=context)
            send_switch_char_cmd(char, active_prompt, current_frame, context=context)

class CEB_Ardy_SceneProperties(bpy.types.PropertyGroup):
    characters: bpy.props.CollectionProperty(
        type=CEB_Ardy_Character
    )
    active_character_index: bpy.props.IntProperty(
        name="Active Character",
        default=0,
        update=update_active_character
    )
    crowds: bpy.props.CollectionProperty(
        type=CEB_Ardy_CrowdItem
    )
    active_crowd_index: bpy.props.IntProperty(
        name="Active Crowd Index",
        default=0,
        update=update_active_crowd
    )
    show_crowd_options: bpy.props.BoolProperty(
        name="Show Crowd Options",
        description="Toggle display of crowd options in the panel",
        default=True
    )
    show_crowd_generation: bpy.props.BoolProperty(
        name="Show Crowd Generation",
        description="Toggle display of Crowd Generation parameters",
        default=True
    )
    show_crowd_management: bpy.props.BoolProperty(
        name="Show Crowd Management",
        description="Toggle display of Crowd Management list and controls",
        default=True
    )
    quantize_4bit: bpy.props.BoolProperty(
        name="4-bit Quantization (bitsandbytes)",
        description="Load text encoder in 4-bit precision to save GPU VRAM (~5.5 GB VRAM instead of ~16 GB)",
        default=True
    )
    import_scale: bpy.props.FloatProperty(
        name="Import Scale",
        description="Scale factor applied to joint coordinates",
        default=1.0,
        min=0.001
    )
    realtime_recording: bpy.props.BoolProperty(
        name="Live Record",
        description="Record the incoming real-time motion stream as keyframes",
        default=True
    )
    mute_previous_nla_layers: bpy.props.BoolProperty(
        name="Mute Previous NLA Layers",
        description="Mute existing NLA tracks/layers on the character when starting a new stream",
        default=False
    )
    source_armature_name: bpy.props.StringProperty(
        name="Source Armature",
        description="Name of the source armature object to retarget from (e.g. GEMX_Armature or MHR_Armature)",
        default="GEMX_Armature"
    )
    mhr_source_armature_name: bpy.props.StringProperty(
        name="Source MHR Armature",
        description="Deprecated alias for source_armature_name",
        default="GEMX_Armature"
    )
    retarget_flip_180: bpy.props.BoolProperty(
        name="180° Facing Correction",
        description="Flip front-to-back rotation orientation by 180° to align facing direction",
        default=True
    )
    realtime_status: bpy.props.StringProperty(
        name="Real-time Status",
        description="Current socket connection status",
        default="Disconnected"
    )
    realtime_port: bpy.props.IntProperty(
        name="Real-time Port",
        description="TCP Port for the socket connection",
        default=9999,
        min=1024,
        max=65535
    )
    overlay_view_mode: bpy.props.EnumProperty(
        name="Overlay View Mode",
        description="Display overlay for active character only or all characters",
        items=[
            ('SELECTED', "Selected Only", "Show overlay for the currently selected character"),
            ('ALL', "All Characters", "Show overlay for all loaded characters in scene")
        ],
        default='SELECTED',
        update=update_overlay_visibility
    )
    show_prompt_overlay: bpy.props.BoolProperty(
        name="Show Overlay in 3D View",
        description="Display real-time prompt overlay in 3D Viewport",
        default=False,
        update=update_overlay_visibility
    )
    ik_control_active: bpy.props.BoolProperty(
        name="IK Control Active",
        description="Whether IK control mode is active for pose editing",
        default=False
    )
    ik_loaded_collection_name: bpy.props.StringProperty(
        name="IK Collection Name",
        description="Name of loaded IK rig collection",
        default=""
    )
    ik_original_armature_name: bpy.props.StringProperty(
        name="IK Original Armature Name",
        description="Name of original pose constraint or character armature being edited",
        default=""
    )
    ik_target_type: bpy.props.EnumProperty(
        name="IK Target Type",
        description="Target type being controlled by IK (Constraint or Character)",
        items=[
            ('CONSTRAINT', "Constraint", "Controlling a pose constraint armature"),
            ('CHARACTER', "Character", "Controlling the main character armature"),
        ],
        default='CONSTRAINT'
    )
    ik_hidden_object_names: bpy.props.StringProperty(
        name="IK Hidden Object Names",
        description="Comma-separated names of objects hidden when starting IK control",
        default=""
    )
    crowd_count: bpy.props.IntProperty(
        name="Crowd Count",
        description="Number of characters in the crowd",
        default=4,
        min=1,
        max=500
    )
    crowd_spacing: bpy.props.FloatProperty(
        name="Spacing (m)",
        description="Distance between crowd characters in meters",
        default=2.5,
        min=0.5,
        max=50.0
    )
    crowd_layout: bpy.props.EnumProperty(
        name="Layout Pattern",
        description="Spatial layout pattern for crowd characters",
        items=[
            ('GRID', "Grid", "Arrange characters in a 2D grid array"),
            ('LINE', "Line", "Arrange characters in a single row line"),
            ('CIRCLE', "Circle", "Arrange characters in a circular ring facing outward"),
            ('RANDOM', "Random Scatter", "Randomly scatter characters with guaranteed minimum separation distance"),
        ],
        default='GRID'
    )
    crowd_offset_waypoints: bpy.props.BoolProperty(
        name="Parallel Trajectories",
        description="Offset waypoints relative to each character's starting position so movement paths run parallel and do not overlap",
        default=True
    )
    crowd_avoid_collisions: bpy.props.BoolProperty(
        name="Avoid Character Collisions",
        description="Automatically insert detour waypoints to steer characters around previously simulated characters and prevent overlapping paths",
        default=True
    )
    crowd_avoid_unsimulated: bpy.props.BoolProperty(
        name="Avoid Standing Locations",
        description="Steer characters around initial standing locations of other characters that have not been simulated yet",
        default=False
    )
    crowd_avoid_radius: bpy.props.FloatProperty(
        name="Safety Buffer (m)",
        description="Minimum distance kept between characters to prevent collision",
        default=0.7,
        min=0.5,
        max=10.0
    )
    crowd_reverse_order: bpy.props.BoolProperty(
        name="Reverse Simulation Order",
        description="Simulate characters starting from the last character in the list down to the first",
        default=True
    )


    crowd_start_frame: bpy.props.IntProperty(
        name="Start Frame",
        description="Start frame for crowd animation",
        default=1,
        min=0
    )
    crowd_end_frame: bpy.props.IntProperty(
        name="End Frame",
        description="End frame for crowd animation",
        default=250,
        min=1
    )




class CEB_OT_AddCharacterEntry(bpy.types.Operator):
    bl_idname = "ceb.add_character_entry"
    bl_label = "Add Character"
    bl_description = "Add a new character configuration"

    def execute(self, context):
        props = context.scene.ceb_ardy
        idx = len(props.characters) + 1
        name = f"Character_{idx}"
        while any(c.name == name for c in props.characters):
            idx += 1
            name = f"Character_{idx}"
        char = props.characters.add()
        char.name = name
        char.model = 'core'
        props.active_character_index = len(props.characters) - 1
        
        # Automatically load the character mesh & armature into the scene
        bpy.ops.ceb.load_character()
        
        tag_redraw_view3d(context)
        return {'FINISHED'}

class CEB_OT_RemoveCharacterEntry(bpy.types.Operator):
    bl_idname = "ceb.remove_character_entry"
    bl_label = "Remove Character"
    bl_description = "Remove the selected character configuration and its Blender objects"

    def execute(self, context):
        props = context.scene.ceb_ardy
        idx = props.active_character_index
        if 0 <= idx < len(props.characters):
            char = props.characters[idx]
            for obj_name in [char.arm_obj_name, char.mesh_obj_name, char.parent_obj_name]:
                if obj_name:
                    obj = bpy.data.objects.get(obj_name)
                    if obj:
                        bpy.data.objects.remove(obj, do_unlink=True)
            for item in char.prompt_schedule:
                remove_waypoint_empty(item)
                remove_pose_constraint_armature(item)
            props.characters.remove(idx)
            props.active_character_index = max(0, idx - 1)
            tag_redraw_view3d(context)
        return {'FINISHED'}

class CEB_OT_AddPromptItem(bpy.types.Operator):
    bl_idname = "ceb.add_prompt_item"
    bl_label = "Add Prompt"
    bl_description = "Add a new prompt schedule item for the active character"

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        item = char.prompt_schedule.add()
        if len(char.prompt_schedule) == 1:
            item.start_frame = 0
        else:
            item.start_frame = context.scene.frame_current
        item.prompt = "walk"
        char.prompt_schedule_index = len(char.prompt_schedule) - 1
        tag_redraw_view3d(context)
        return {'FINISHED'}

class CEB_OT_RemovePromptItem(bpy.types.Operator):
    bl_idname = "ceb.remove_prompt_item"
    bl_label = "Remove Prompt"
    bl_description = "Remove the selected prompt schedule item"

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        idx = char.prompt_schedule_index
        if 0 <= idx < len(char.prompt_schedule):
            item = char.prompt_schedule[idx]
            remove_waypoint_empty(item)
            remove_pose_constraint_armature(item)
            char.prompt_schedule.remove(idx)
            char.prompt_schedule_index = max(0, idx - 1)
            tag_redraw_view3d(context)
            send_waypoints_to_bridge(context)
            send_pose_constraints_to_bridge(context)
        return {'FINISHED'}

class CEB_OT_ClearPromptItems(bpy.types.Operator):
    bl_idname = "ceb.clear_prompt_items"
    bl_label = "Clear All Prompts"
    bl_description = "Remove all items from the active character's prompt schedule"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        for item in char.prompt_schedule:
            remove_waypoint_empty(item)
            remove_pose_constraint_armature(item)
        char.prompt_schedule.clear()
        char.prompt_schedule_index = 0
        tag_redraw_view3d(context)
        send_waypoints_to_bridge(context)
        send_pose_constraints_to_bridge(context)
        self.report({'INFO'}, f"Cleared prompt schedule items for {char.name}.")
        return {'FINISHED'}

class CEB_OT_MovePromptItem(bpy.types.Operator):
    bl_idname = "ceb.move_prompt_item"
    bl_label = "Move Prompt"
    bl_description = "Move selected prompt item up or down"

    direction: bpy.props.EnumProperty(
        items=[('UP', 'Up', ''), ('DOWN', 'Down', '')]
    )

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        idx = char.prompt_schedule_index
        schedule = char.prompt_schedule
        if self.direction == 'UP' and idx > 0:
            schedule.move(idx, idx - 1)
            char.prompt_schedule_index -= 1
        elif self.direction == 'DOWN' and idx < len(schedule) - 1:
            schedule.move(idx, idx + 1)
            char.prompt_schedule_index += 1
        tag_redraw_view3d(context)
        return {'FINISHED'}

class CEB_OT_SortPromptItems(bpy.types.Operator):
    bl_idname = "ceb.sort_prompt_items"
    bl_label = "Sort Schedule by Frame"
    bl_description = "Reorder prompt schedule items chronologically by start frame"

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        schedule = char.prompt_schedule
        n = len(schedule)
        if n <= 1:
            return {'FINISHED'}

        # In-place selection sort using schedule.move(min_idx, i)
        # This reorders collection items without clearing/re-adding or duplicating objects
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if schedule[j].start_frame < schedule[min_idx].start_frame:
                    min_idx = j
            if min_idx != i:
                schedule.move(min_idx, i)

        # Sync object names for all items after reordering
        for item in schedule:
            sync_prompt_item_object_names(item, context)

        char.prompt_schedule_index = 0
        tag_redraw_view3d(context)
        send_waypoints_to_bridge(context)
        send_pose_constraints_to_bridge(context)
        self.report({'INFO'}, f"Prompt schedule reordered by start frame for {char.name}.")
        return {'FINISHED'}

class CEB_OT_RemoveWaypoint(bpy.types.Operator):
    bl_idname = "ceb.remove_waypoint"
    bl_label = "Remove Waypoint"
    bl_description = "Remove 3D Waypoint target and delete linked Empty object"

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        idx = char.prompt_schedule_index
        if 0 <= idx < len(char.prompt_schedule):
            item = char.prompt_schedule[idx]
            remove_waypoint_empty(item)
            item.has_waypoint = False
            tag_redraw_view3d(context)
            send_waypoints_to_bridge(context)
            self.report({'INFO'}, "Waypoint removed.")
        return {'FINISHED'}

class CEB_OT_AddPoseConstraint(bpy.types.Operator):
    bl_idname = "ceb.add_pose_constraint"
    bl_label = "Add Pose Constraint"
    bl_description = "Add a full-body pose constraint by duplicating the character armature for target posing"

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        item = char.prompt_schedule.add()
        item.start_frame = 0 if len(char.prompt_schedule) == 1 else context.scene.frame_current
        item.prompt = ""
        item.has_pose_constraint = True

        arm_obj = get_or_create_pose_constraint_armature(context, item, copy_current_pose=True)

        char.prompt_schedule_index = len(char.prompt_schedule) - 1
        tag_redraw_view3d(context)
        send_pose_constraints_to_bridge(context)

        if arm_obj:
            bpy.ops.object.select_all(action='DESELECT')
            arm_obj.select_set(True)
            context.view_layer.objects.active = arm_obj
            self.report({'INFO'}, f"Added Pose Constraint armature '{arm_obj.name}' at frame {item.start_frame}")
        else:
            self.report({'WARNING'}, "Added Pose Constraint item, but no character armature was found to duplicate.")

        return {'FINISHED'}

class CEB_OT_CapturePoseConstraint(bpy.types.Operator):
    bl_idname = "ceb.capture_pose_constraint"
    bl_label = "Capture Current Pose Constraint"
    bl_description = "Capture the selected character's current pose and set it as a pose constraint target"

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            self.report({'ERROR'}, "No active character selected.")
            return {'CANCELLED'}

        arm_obj = None
        if char.arm_obj_name:
            arm_obj = bpy.data.objects.get(char.arm_obj_name)
        if not arm_obj and context.active_object and context.active_object.type == 'ARMATURE' and not context.active_object.name.startswith("ARDY_Pose_"):
            arm_obj = context.active_object
        if not arm_obj:
            clean_name = char.name.replace(" ", "_")
            arm_obj = bpy.data.objects.get(f"{clean_name}_Armature")

        if not arm_obj:
            self.report({'WARNING'}, f"Character armature for '{char.name}' not found.")
            return {'CANCELLED'}

        idx = char.prompt_schedule_index
        item = None
        if 0 <= idx < len(char.prompt_schedule) and getattr(char.prompt_schedule[idx], "has_pose_constraint", False):
            item = char.prompt_schedule[idx]
        else:
            item = char.prompt_schedule.add()
            item.start_frame = context.scene.frame_current
            item.prompt = ""
            item.has_pose_constraint = True
            char.prompt_schedule_index = len(char.prompt_schedule) - 1

        pose_arm = get_or_create_pose_constraint_armature(context, item, copy_current_pose=True)

        if pose_arm and arm_obj.pose and pose_arm.pose:
            pose_arm.matrix_world = arm_obj.matrix_world.copy()
            for src_bone in arm_obj.pose.bones:
                tgt_bone = pose_arm.pose.bones.get(src_bone.name)
                if tgt_bone:
                    tgt_bone.rotation_mode = src_bone.rotation_mode
                    tgt_bone.location = src_bone.location.copy()
                    tgt_bone.rotation_quaternion = src_bone.rotation_quaternion.copy()
                    tgt_bone.rotation_euler = src_bone.rotation_euler.copy()
                    tgt_bone.rotation_axis_angle = list(src_bone.rotation_axis_angle)
                    tgt_bone.scale = src_bone.scale.copy()

        tag_redraw_view3d(context)
        send_pose_constraints_to_bridge(context)

        self.report({'INFO'}, f"Captured pose of '{char.name}' as constraint at frame {item.start_frame}")
        return {'FINISHED'}

class CEB_OT_RemovePoseConstraint(bpy.types.Operator):
    bl_idname = "ceb.remove_pose_constraint"
    bl_label = "Remove Pose Constraint"
    bl_description = "Remove full-body Pose Constraint target and delete linked ghost armature object"

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        idx = char.prompt_schedule_index
        if 0 <= idx < len(char.prompt_schedule):
            item = char.prompt_schedule[idx]
            remove_pose_constraint_armature(item)
            item.has_pose_constraint = False
            tag_redraw_view3d(context)
            send_pose_constraints_to_bridge(context)
            self.report({'INFO'}, "Pose constraint removed.")
        return {'FINISHED'}

class CEB_OT_SelectPromptItem(bpy.types.Operator):
    bl_idname = "ceb.select_prompt_item"
    bl_label = "Select Prompt Item"
    bl_description = "Select this prompt schedule item, jump to its start frame, and select its waypoint object"

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            return {'CANCELLED'}
        if 0 <= self.index < len(char.prompt_schedule):
            char.prompt_schedule_index = self.index
            item = char.prompt_schedule[self.index]
            
            context.scene.frame_set(item.start_frame)
            
            if getattr(item, "has_waypoint", False) and getattr(item, "waypoint_object_name", ""):
                obj = bpy.data.objects.get(item.waypoint_object_name)
                if obj:
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
            elif getattr(item, "has_pose_constraint", False) and getattr(item, "pose_armature_name", ""):
                obj = bpy.data.objects.get(item.pose_armature_name)
                if obj:
                    bpy.ops.object.select_all(action='DESELECT')
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    
            tag_redraw_view3d(context)
        return {'FINISHED'}

def get_or_create_waypoint_empty(context, item):
    obj_name = item.waypoint_object_name
    obj = bpy.data.objects.get(obj_name) if obj_name else None
    
    if not obj:
        char = get_active_character(context)
        prefix = get_char_prefix(char)
        base_name = f"{prefix}_Ardy_waypoint_F{item.start_frame}"
        obj_name = base_name
        idx = 1
        while bpy.data.objects.get(obj_name):
            obj_name = f"{base_name}_{idx}"
            idx += 1
            
        empty_data = None
        obj = bpy.data.objects.new(obj_name, empty_data)
        obj.empty_display_type = 'SINGLE_ARROW'
        obj.empty_display_size = 0.5
        obj.show_name = True
        
        col_name = "ARDY_Waypoints"
        collection = bpy.data.collections.get(col_name)
        if not collection:
            collection = bpy.data.collections.new(col_name)
            context.scene.collection.children.link(collection)
        collection.objects.link(obj)
        
        item.waypoint_object_name = obj.name

    target_co = mathutils.Vector(item.waypoint_co)
    if obj.parent:
        if hasattr(context, "view_layer") and context.view_layer:
            context.view_layer.update()
        obj.location = obj.parent.matrix_world.inverted() @ target_co
        obj.matrix_world.translation = target_co
    else:
        obj.location = target_co
    return obj

@persistent
def ardy_depsgraph_sync_waypoints(scene, depsgraph=None):
    if not hasattr(scene, "ceb_ardy"):
        return
    props = scene.ceb_ardy
    for char in props.characters:
        for item in char.prompt_schedule:
            if item.has_waypoint and item.waypoint_object_name:
                obj = bpy.data.objects.get(item.waypoint_object_name)
                if obj:
                    world_co = obj.matrix_world.to_translation()
                    loc_vec = mathutils.Vector(item.waypoint_co)
                    if (world_co - loc_vec).length > 1e-4:
                        item.waypoint_co = world_co

class CEB_OT_AddWaypoint(bpy.types.Operator):
    bl_idname = "ceb.add_waypoint"
    bl_label = "Add Waypoint in 3D View"
    bl_description = "Click in the 3D Viewport to place a new Waypoint for the active character"

    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            region = context.region
            rv3d = context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)

            from bpy_extras.view3d_utils import region_2d_to_location_3d, region_2d_to_vector_3d, region_2d_to_origin_3d
            
            origin = region_2d_to_origin_3d(region, rv3d, coord)
            direction = region_2d_to_vector_3d(region, rv3d, coord)

            if abs(direction.z) > 1e-6:
                t = -origin.z / direction.z
                target_co = origin + t * direction
            else:
                target_co = region_2d_to_location_3d(region, rv3d, coord, (0, 0, 0))

            char = get_active_character(context)
            if not char:
                return {'CANCELLED'}
            item = char.prompt_schedule.add()
            item.start_frame = 0 if len(char.prompt_schedule) == 1 else context.scene.frame_current
            item.prompt = ""
            item.has_waypoint = True
            item.waypoint_co = target_co

            get_or_create_waypoint_empty(context, item)

            char.prompt_schedule_index = len(char.prompt_schedule) - 1
            tag_redraw_view3d(context)
            send_waypoints_to_bridge(context)
            self.report({'INFO'}, f"Placed Waypoint for {char.name} at ({target_co.x:.2f}, {target_co.y:.2f}, {target_co.z:.2f})")
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.report({'INFO'}, "Waypoint placement cancelled.")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.space_data and context.space_data.type == 'VIEW_3D':
            context.window_manager.modal_handler_add(self)
            self.report({'INFO'}, "Click in 3D Viewport to place Waypoint (Esc to cancel)")
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "Active space must be a 3D Viewport")
            return {'CANCELLED'}

def draw_round_rect_2d(x, y, width, height, color):
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    except Exception:
        try:
            shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        except Exception:
            return
    shader.bind()
    shader.uniform_float("color", color)
    
    vertices = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height)
    ]
    indices = [(0, 1, 2), (0, 2, 3)]
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    batch.draw(shader)

def draw_waypoints_3d_view(self, context):
    if context is None:
        context = bpy.context
    if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
        return
    props = context.scene.ceb_ardy
    if not props.show_prompt_overlay:
        return

    try:
        shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
    except Exception:
        try:
            shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        except Exception:
            return

    shader.bind()
    font_id = 0

    try:
        gpu.state.blend_set('ALPHA')
    except Exception:
        pass

    chars_to_draw = []
    if props.overlay_view_mode == 'SELECTED':
        active_c = get_active_character(context)
        if active_c:
            chars_to_draw.append(active_c)
    else:
        chars_to_draw = list(props.characters)

    palette = [
        (0.0, 0.85, 1.0, 0.9),
        (1.0, 0.45, 0.2, 0.9),
        (0.2, 0.95, 0.4, 0.9),
        (0.9, 0.3, 0.9, 0.9),
    ]

    for c_idx, char in enumerate(chars_to_draw):
        color = palette[c_idx % len(palette)]
        for item in char.prompt_schedule:
            if not item.enabled or not getattr(item, "has_waypoint", False):
                continue

            co = mathutils.Vector(item.waypoint_co)
            ground_co = mathutils.Vector((co.x, co.y, 0.0))
            top_co = mathutils.Vector((co.x, co.y, co.z + 0.6))

            vertices = [(ground_co.x, ground_co.y, ground_co.z), (top_co.x, top_co.y, top_co.z)]
            
            shader.uniform_float("color", color)
            batch = batch_for_shader(shader, 'LINES', {"pos": vertices})
            batch.draw(shader)

            region = context.region
            rv3d = context.region_data
            if region and rv3d:
                from bpy_extras.view3d_utils import location_3d_to_region_2d
                screen_pos = location_3d_to_region_2d(region, rv3d, top_co)
                if screen_pos:
                    lbl_text = f"📍 [{char.name}] F{item.start_frame}: Waypoint ({co.x:.1f}, {co.y:.1f}, {co.z:.1f})"
                    try:
                        blf.size(font_id, 11)
                        blf.color(font_id, color[0], color[1], color[2], 1.0)
                        blf.position(font_id, int(screen_pos.x) + 8, int(screen_pos.y) + 4, 0)
                        blf.draw(font_id, lbl_text)
                    except Exception:
                        pass

def draw_prompt_overlay_px(self, context):
    if context is None:
        context = bpy.context
    if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
        return
    props = context.scene.ceb_ardy
    if not props.show_prompt_overlay:
        return

    region = context.region
    if not region or region.width < 100 or region.height < 100:
        return

    chars_to_draw = []
    if props.overlay_view_mode == 'SELECTED':
        c = get_active_character(context)
        if c:
            chars_to_draw.append(c)
    else:
        chars_to_draw = list(props.characters)

    if not chars_to_draw:
        return

    current_frame = context.scene.frame_current
    scene_start = context.scene.frame_start
    scene_end = context.scene.frame_end
    
    font_id = 0
    card_w = min(680, max(380, int(region.width * 0.65)))
    card_h = 108
    base_y = 35

    for c_idx, char in enumerate(chars_to_draw):
        y = base_y + c_idx * (card_h + 10)
        sorted_schedule = sorted(char.prompt_schedule, key=lambda x: x.start_frame)
        enabled_schedule = [item for item in sorted_schedule if item.enabled]

        if not enabled_schedule and not char.realtime_prompt:
            continue

        active_item = None
        active_prompt_text = char.realtime_prompt if char.realtime_prompt else "None"
        
        for item in sorted_schedule:
            if item.enabled and current_frame >= item.start_frame:
                active_item = item
                active_prompt_text = item.prompt

        max_sched_frame = max([item.start_frame for item in sorted_schedule], default=scene_end)
        frame_min = scene_start
        frame_max = max(scene_end, max_sched_frame + 20)
        total_frames = max(1, frame_max - frame_min)

        x = int((region.width - card_w) / 2)

        try:
            gpu.state.blend_set('ALPHA')
        except Exception:
            pass

        draw_round_rect_2d(x, y, card_w, card_h, (0.08, 0.10, 0.15, 0.88))
        draw_round_rect_2d(x, y + card_h - 4, card_w, 4, (0.15, 0.65, 0.95, 0.9))

        try:
            blf.size(font_id, 10)
            blf.color(font_id, 0.55, 0.65, 0.75, 1.0)
            blf.position(font_id, x + 16, y + card_h - 22, 0)
            blf.draw(font_id, f"PROMPT TIMELINE ({char.name})")
        except Exception:
            pass

        active_str = f"► Active: \"{active_prompt_text}\" (Frame {current_frame})"
        try:
            blf.size(font_id, 12)
            blf.color(font_id, 0.2, 0.95, 0.45, 1.0)
            blf.position(font_id, x + 240, y + card_h - 22, 0)
            blf.draw(font_id, active_str)
        except Exception:
            pass

        track_padding = 16
        track_x = x + track_padding
        track_w = card_w - (track_padding * 2)
        track_y = y + 44
        track_h = 18

        draw_round_rect_2d(track_x, track_y, track_w, track_h, (0.15, 0.18, 0.24, 0.9))

        def frame_to_x(f):
            norm = (f - frame_min) / total_frames
            norm = max(0.0, min(1.0, norm))
            return track_x + norm * track_w

        if sorted_schedule:
            for i, item in enumerate(sorted_schedule):
                f_start = item.start_frame
                f_end = sorted_schedule[i + 1].start_frame if i + 1 < len(sorted_schedule) else frame_max

                x1 = frame_to_x(f_start)
                x2 = frame_to_x(f_end)
                seg_w = max(2.0, x2 - x1)

                is_active = (item == active_item)
                if is_active:
                    col = (0.1, 0.75, 0.95, 0.85) if item.enabled else (0.4, 0.5, 0.6, 0.5)
                else:
                    col = (0.22, 0.32, 0.45, 0.7) if item.enabled else (0.14, 0.16, 0.2, 0.4)

                draw_round_rect_2d(int(x1), track_y + 2, int(seg_w), track_h - 4, col)
                draw_round_rect_2d(int(x1), track_y - 2, 2, track_h + 4, (1.0, 1.0, 1.0, 0.8 if item.enabled else 0.3))

                if getattr(item, "has_waypoint", False):
                    px_center = int(x1)
                    py_center = track_y + track_h // 2
                    draw_round_rect_2d(px_center - 3, py_center - 4, 6, 8, (0.0, 0.95, 1.0, 1.0))
                    lbl_str = f"📍 F{f_start}: {item.prompt}"
                else:
                    lbl_str = f"F{f_start}: {item.prompt}"

                try:
                    blf.size(font_id, 10)
                    if is_active:
                        blf.color(font_id, 1.0, 0.85, 0.3, 1.0)
                    elif item.enabled:
                        blf.color(font_id, 0.8, 0.85, 0.9, 0.9)
                    else:
                        blf.color(font_id, 0.5, 0.5, 0.5, 0.6)

                    lbl_w, _ = blf.dimensions(font_id, lbl_str)
                    lbl_x = max(track_x, min(int(x1), track_x + track_w - int(lbl_w)))
                    blf.position(font_id, lbl_x, y + 22, 0)
                    blf.draw(font_id, lbl_str)
                except Exception:
                    pass

        playhead_x = int(frame_to_x(current_frame))
        draw_round_rect_2d(playhead_x - 1, track_y - 6, 3, track_h + 12, (1.0, 0.55, 0.1, 1.0))
        draw_round_rect_2d(playhead_x - 4, track_y + track_h + 4, 9, 6, (1.0, 0.65, 0.15, 1.0))

    try:
        gpu.state.blend_set('NONE')
    except Exception:
        pass

@persistent
def ardy_frame_change_handler(scene):
    if not hasattr(scene, "ceb_ardy"):
        return
    props = scene.ceb_ardy
    current_frame = scene.frame_current
    
    char = get_active_character()
    if char:
        sorted_schedule = sorted([item for item in char.prompt_schedule if item.enabled], key=lambda x: x.start_frame)
        active_prompt = None
        for item in sorted_schedule:
            if current_frame >= item.start_frame:
                active_prompt = item.prompt
                
        if active_prompt and active_prompt != char.realtime_prompt:
            char.realtime_prompt = active_prompt

    tag_redraw_view3d()

def get_ardy_paths(context):
    package_name = __package__ if __package__ else "CEB_Ardy"
    prefs = context.preferences.addons.get(package_name)
    if not prefs or not prefs.preferences.ardy_path:
        return None, "Please configure the Portable Path in the Addon Preferences."
    
    selected_path = os.path.abspath(bpy.path.abspath(prefs.preferences.ardy_path))
    if not os.path.isdir(selected_path):
        return None, f"Configured path is not a directory: {selected_path}"
    
    python_exe = None
    ardy_dir = None
    
    # CASE 1: Selected path contains python.exe directly (e.g. D:\_Code\Meu\CEB_Ardy_prj\Portable_Ardy\python-3.11.9-embed-amd64)
    if os.path.exists(os.path.join(selected_path, "python.exe")):
        python_exe = os.path.join(selected_path, "python.exe")
        parent_dir = os.path.dirname(selected_path)
        candidate_ardy = os.path.join(parent_dir, "ardy")
        if os.path.isdir(candidate_ardy):
            ardy_dir = candidate_ardy
            
    # CASE 2: Selected path is the parent directory containing both python folder and ardy folder (e.g. D:\_Code\Meu\CEB_Ardy_prj\Portable_Ardy)
    if not python_exe or not ardy_dir:
        candidate_ardy = os.path.join(selected_path, "ardy")
        if os.path.isdir(candidate_ardy):
            # Find python.exe inside any subdirectories of selected_path
            for item in os.listdir(selected_path):
                item_path = os.path.join(selected_path, item)
                if os.path.isdir(item_path):
                    candidate_py = os.path.join(item_path, "python.exe")
                    if os.path.exists(candidate_py):
                        python_exe = candidate_py
                        ardy_dir = candidate_ardy
                        break
                        
    # CASE 3: Fallback (original behaviour if folder itself is ardy and has venv)
    if not python_exe or not ardy_dir:
        candidate_py = os.path.join(selected_path, ".venv", "Scripts", "python.exe")
        if os.path.exists(candidate_py):
            python_exe = candidate_py
            ardy_dir = selected_path
        else:
            candidate_py = os.path.join(selected_path, "venv", "Scripts", "python.exe")
            if os.path.exists(candidate_py):
                python_exe = candidate_py
                ardy_dir = selected_path

    if not python_exe:
        return None, "Could not find python.exe. Please select the python folder."
        
    if not ardy_dir:
        return None, "Could not locate the 'ardy' directory next to the Python path."
        
    generate_script = os.path.join(ardy_dir, "scripts", "generate.py")
    if not os.path.exists(generate_script):
        return None, f"Could not find 'scripts/generate.py' in ARDY path: {ardy_dir}"
        
    return {
        "ardy_dir": ardy_dir,
        "python_exe": python_exe,
        "generate_script": generate_script,
        "run_demo_script": os.path.join(ardy_dir, "scripts", "run_demo.py"),
        "run_server_script": os.path.join(ardy_dir, "scripts", "run_text_encoder_server.py"),
    }, None

soma30_names = [
    'Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'Jaw', 'LeftEye', 'RightEye',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumbEnd', 'LeftHandMiddleEnd',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumbEnd', 'RightHandMiddleEnd',
    'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase'
]

smpl24_names = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
    'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

smpl22_names = smpl24_names[:22]

ARDY_TO_BLENDER_MATRIX = mathutils.Matrix([
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [ 0.0,  1.0,  0.0]
])

def ardy_pos_to_blender(pos_ardy, scale=1.0):
    """Convert ARDY position (X-Left, Y-Up, Z-Forward) to Blender position (X-Right, Y-Forward, Z-Up)."""
    return mathutils.Vector((-float(pos_ardy[0]) * scale, float(pos_ardy[2]) * scale, float(pos_ardy[1]) * scale))

def ardy_rot_to_blender(R_a):
    """Convert ARDY 3x3 rotation matrix to Blender 3x3 rotation matrix using M = [[-1,0,0],[0,0,1],[0,1,0]]."""
    m = mathutils.Matrix.Identity(3)
    if R_a is not None:
        m[0][0] =  float(R_a[0][0]); m[0][1] = -float(R_a[0][2]); m[0][2] = -float(R_a[0][1])
        m[1][0] = -float(R_a[2][0]); m[1][1] =  float(R_a[2][2]); m[1][2] =  float(R_a[2][1])
        m[2][0] = -float(R_a[1][0]); m[2][1] =  float(R_a[1][2]); m[2][2] =  float(R_a[1][1])
    return m

def ardy_transform_to_blender_matrix(pos_ardy, R_a, scale=1.0):
    """Construct a 4x4 Blender matrix from ARDY position and 3x3 rotation."""
    t_b = ardy_pos_to_blender(pos_ardy, scale)
    R_b = ardy_rot_to_blender(R_a)
    mat = R_b.to_4x4()
    mat.translation = t_b
    return mat

def get_skin_path(paths, J):
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    skin_filename = "skin_core.npz" if J == 27 else "skin_standard.npz"
    skel_dir = "cskel27" if J == 27 else "somaskel77"

    candidate_paths = [
        os.path.join(addon_dir, "data", skin_filename),
        os.path.join(addon_dir, "data", skel_dir, "skin_standard.npz"),
        r"D:\_Code\Meu\CEB_Ardy_prj\Portable_Ardy\ardy\ardy\assets\skeletons\cskel27\skin_standard.npz" if J == 27 else r"D:\_Code\Meu\CEB_Ardy_prj\Portable_Ardy\ardy\ardy\assets\skeletons\somaskel77\skin_standard.npz",
        r"D:\_Code\Meu\CEB_Ardy_prj\Portable_Ardy\python-3.11.9-embed-amd64\Lib\site-packages\ardy\assets\skeletons\cskel27\skin_standard.npz" if J == 27 else r"D:\_Code\Meu\CEB_Ardy_prj\Portable_Ardy\python-3.11.9-embed-amd64\Lib\site-packages\ardy\assets\skeletons\somaskel77\skin_standard.npz",
    ]
    if paths and isinstance(paths, dict) and "ardy_dir" in paths and paths["ardy_dir"]:
        rel = os.path.join("ardy", "assets", "skeletons", skel_dir, "skin_standard.npz")
        candidate_paths.append(os.path.join(paths["ardy_dir"], rel))

    for path in candidate_paths:
        if os.path.exists(path):
            return path
            
    if paths and isinstance(paths, dict) and "ardy_dir" in paths and paths["ardy_dir"]:
        return os.path.join(paths["ardy_dir"], "ardy", "assets", "skeletons", skel_dir, "skin_standard.npz")
    return candidate_paths[0]

def setup_soma_skin(context, parent_obj, J, scale, paths, char_name=None, skin_path=None):
    try:
        import numpy as np
    except ImportError:
        return None, None
        
    if not skin_path:
        skin_path = get_skin_path(paths, J)
        
    if not skin_path or not os.path.exists(skin_path):
        print(f"[CEB Ardy] Skin path not found: {skin_path}")
        return None, None
        
    try:
        skin_data = np.load(skin_path)
    except Exception as e:
        print(f"[CEB Ardy] Failed to load skin npz ({skin_path}): {e}")
        return None, None
        
    bind_vertices = skin_data["bind_vertices"]
    faces = skin_data["faces"]
    bind_rig_transform = skin_data["bind_rig_transform"]
    rig_joint_names = [str(n) for n in skin_data["rig_joint_names"]]
    lbs_indices = skin_data["lbs_indices"]
    lbs_weights = skin_data["lbs_weights"]
    rig_joint_connections = skin_data["rig_joint_connections"]
    
    if parent_obj:
        parent_obj.rotation_euler = (0, 0, 0)

    clean_prefix = char_name.replace(" ", "_") if char_name else ("Core" if J == 27 else "SOMA")

    # 1. Create Mesh
    mesh_data = bpy.data.meshes.new(name=f"{clean_prefix}_Skin_Mesh")
    mesh_obj = bpy.data.objects.new(f"{clean_prefix}_Skin", mesh_data)
    context.scene.collection.objects.link(mesh_obj)
    
    verts = [tuple(ardy_pos_to_blender(v, scale)) for v in bind_vertices]
    faces_list = [tuple(f) for f in faces]
    mesh_data.from_pydata(verts, [], faces_list)
    mesh_data.update()
    
    # 2. Create Armature
    arm_data = bpy.data.armatures.new(f"{clean_prefix}_Armature_Data")
    arm_obj = bpy.data.objects.new(f"{clean_prefix}_Armature", arm_data)
    context.scene.collection.objects.link(arm_obj)
    if parent_obj:
        arm_obj.parent = parent_obj

    # Parent mesh directly to the armature
    mesh_obj.parent = arm_obj
    
    # Add Armature Modifier
    arm_mod = mesh_obj.modifiers.new(name=f"{clean_prefix}_Armature_Mod", type='ARMATURE')
    arm_mod.object = arm_obj
    
    # 3. Create EditBones anchored to exact joint heads and tails
    context.view_layer.update()
    original_active = context.view_layer.objects.active
    context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    
    edit_bones = arm_obj.data.edit_bones
    
    bind_matrices_b = []
    for i in range(len(rig_joint_names)):
        pos_i = bind_rig_transform[i, :3, 3]
        rot_i = bind_rig_transform[i, :3, :3]
        bind_matrices_b.append(ardy_transform_to_blender_matrix(pos_i, rot_i, scale))

    # Map each parent joint index to its primary child joint index
    child_of_joint = {}
    for p_idx, c_idx in rig_joint_connections:
        if p_idx not in child_of_joint:
            child_of_joint[p_idx] = c_idx

    for i in range(len(rig_joint_names)):
        b_name = rig_joint_names[i]
        bone = edit_bones.new(name=b_name)
        
        h_pos = bind_matrices_b[i].to_translation()
        if i in child_of_joint:
            c_idx = child_of_joint[i]
            t_pos = bind_matrices_b[c_idx].to_translation()
            dist = (t_pos - h_pos).length
            if dist < 0.001 * scale:
                t_pos = h_pos + bind_matrices_b[i].to_3x3() @ mathutils.Vector((0, 0.05 * scale, 0))
        else:
            t_pos = h_pos + bind_matrices_b[i].to_3x3() @ mathutils.Vector((0, 0.05 * scale, 0))
            
        bone.head = h_pos
        bone.tail = t_pos

    # Set parent relationships
    for p_idx, c_idx in rig_joint_connections:
        p_name = rig_joint_names[p_idx]
        c_name = rig_joint_names[c_idx]
        bone = edit_bones.get(c_name)
        parent_bone = edit_bones.get(p_name)
        if bone and parent_bone:
            bone.parent = parent_bone
            
    bpy.ops.object.mode_set(mode='OBJECT')
    if original_active and original_active.name in context.view_layer.objects:
        context.view_layer.objects.active = original_active
    
    # 4. Skin the Mesh (vertex weights for all W weight slots)
    for name in rig_joint_names:
        mesh_obj.vertex_groups.new(name=name)
        
    num_weights = lbs_indices.shape[1]
    for v_idx in range(len(bind_vertices)):
        for i in range(num_weights):
            joint_idx = lbs_indices[v_idx, i]
            weight = lbs_weights[v_idx, i]
            if weight > 0.0001:
                joint_name = rig_joint_names[joint_idx]
                v_group = mesh_obj.vertex_groups[joint_name]
                v_group.add([v_idx], float(weight), 'REPLACE')
                
    return arm_obj, rig_joint_names


def apply_relaxed_idle_pose(arm_obj):
    """
    Apply a natural relaxed idle pose to an ARDY armature after loading.
    Rotates the upper arms down from the default T-pose to a comfortable
    at-the-sides position, matching what the ARDY viser web app displays
    as the initial idle/neutral state.

    The cskel27 arm bones are oriented with local Y along the bone chain
    and local X pointing "up" in the bind pose, so rotating around local X
    swings the arms downward (positive X = down for RightArm, negative X for LeftArm).

    Only affects pose bones – does not create any keyframes.
    """
    import math

    # Maps: bone_name -> (axis, angle_degrees) in LOCAL bone space.
    # Y runs along the bone; X is perpendicular and controls up/down swing.
    # Positive X on RightArm swings it downward; negative X on LeftArm swings it downward.
    RELAXED_ROTATIONS = {
        # Right arm chain
        # "RightShoulder":  ('X',  10.0),   # slight downward roll at shoulder
        "RightArm":       ('X',  -80.0),   # swing upper arm down alongside body
        # "RightForeArm":   ('Z',   5.0),   # subtle elbow bend outward
        # Left arm chain (local X is mirrored, so positive = down here too)
        # "LeftShoulder":   ('X',  10.0),   # slight downward roll at shoulder
        "LeftArm":        ('X',  -80.0),   # swing upper arm down alongside body
        # "LeftForeArm":    ('Z',  -5.0),   # subtle elbow bend outward
    }

    if arm_obj is None or arm_obj.type != 'ARMATURE':
        return

    for bone_name, (axis, angle_deg) in RELAXED_ROTATIONS.items():
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            continue
        pbone.rotation_mode = 'XYZ'
        angle_rad = math.radians(angle_deg)
        rot = [0.0, 0.0, 0.0]
        rot["XYZ".index(axis)] = angle_rad
        pbone.rotation_euler = mathutils.Euler(rot, 'XYZ')




def apply_soma_pose(arm_obj, rig_joint_names, bind_rig_transform,
                    joints_t, rot_mats_t, s77_to_s30, J, scale,
                    record_keys=False, frame_num=0):
    """
    Apply one frame of pose to the armature using pure 3x3 rotational kinematics.
    Root bone (Hips) translates and rotates; child bones use pure local quaternions with zero translation noise.
    Pre-calculates target rotation matrices for all joints in the frame to avoid reading stale bone.parent.matrix.
    """

    num_joints = len(rig_joint_names)
    name_to_idx = {name: idx for idx, name in enumerate(rig_joint_names)}

    # Step 1: Pre-calculate posed matrices and target rotations for all joints in this frame
    R_target_dict = {}
    posed_mat_b_dict = {}

    for j_idx in range(num_joints):
        if J == num_joints:
            j = j_idx
        elif J == 30:
            j = s77_to_s30.get(j_idx, 0)
        else:
            j = j_idx

        pos_a = joints_t[j]
        rot_a = rot_mats_t[j] if rot_mats_t is not None else None

        posed_mat_b = ardy_transform_to_blender_matrix(pos_a, rot_a, scale)
        bind_mat_b  = ardy_transform_to_blender_matrix(bind_rig_transform[j_idx, :3, 3], bind_rig_transform[j_idx, :3, :3], scale)

        posed_mat_b_dict[j_idx] = (posed_mat_b, bind_mat_b)

        c_name = rig_joint_names[j_idx]
        bone = arm_obj.pose.bones.get(c_name)
        if bone:
            R_posed = posed_mat_b.to_3x3()
            R_bind  = bind_mat_b.to_3x3()
            R_edit  = bone.bone.matrix_local.to_3x3()
            R_target_dict[j_idx] = R_posed @ R_bind.inverted() @ R_edit

    # Step 2: Apply poses to armature bones sequentially
    for j_idx in range(num_joints):
        c_name = rig_joint_names[j_idx]
        bone = arm_obj.pose.bones.get(c_name)
        if not bone:
            continue

        posed_mat_b, bind_mat_b = posed_mat_b_dict[j_idx]
        is_root = (j_idx == 0 or bone.parent is None)

        if is_root:
            target_matrix = arm_obj.matrix_world.inverted() @ posed_mat_b @ bind_mat_b.inverted() @ bone.bone.matrix_local
            bone.matrix = target_matrix
        else:
            bone.rotation_mode = 'QUATERNION'
            bone.location = mathutils.Vector((0.0, 0.0, 0.0))

            R_target = R_target_dict[j_idx]
            R_edit   = bone.bone.matrix_local.to_3x3()
            R_parent_edit = bone.parent.bone.matrix_local.to_3x3()

            p_idx = name_to_idx.get(bone.parent.name, None)
            if p_idx is not None and p_idx in R_target_dict:
                R_parent = R_target_dict[p_idx]
            else:
                R_parent = bone.parent.matrix.to_3x3()

            R_rest_local = R_parent_edit.inverted() @ R_edit
            R_expected_parent = R_parent @ R_rest_local

            R_local_pose = R_expected_parent.inverted() @ R_target
            bone.rotation_quaternion = R_local_pose.to_quaternion()

        if record_keys:
            if is_root:
                bone.keyframe_insert(data_path="location", frame=frame_num)
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)


def import_ardy_npz(filepath, context, op=None):
    if not os.path.exists(filepath):
        msg = f"File not found: {filepath}"
        if op:
            op.report({'ERROR'}, msg)
        return {'CANCELLED'}

    try:
        import numpy as np
    except ImportError:
        msg = "NumPy is not available in Blender's Python environment."
        if op:
            op.report({'ERROR'}, msg)
        return {'CANCELLED'}

    try:
        data = np.load(filepath)
    except Exception as e:
        msg = f"Failed to load NPZ file: {e}"
        if op:
            op.report({'ERROR'}, msg)
        return {'CANCELLED'}

    posed_joints = None
    if 'posed_joints' in data:
        posed_joints = data['posed_joints']
    else:
        for key in data.keys():
            arr = data[key]
            if hasattr(arr, 'shape') and len(arr.shape) == 3 and arr.shape[2] == 3:
                posed_joints = arr
                break
                
    if posed_joints is None:
        msg = "Could not find a valid joint positions array (shape [T, J, 3]) in the NPZ file."
        if op:
            op.report({'ERROR'}, msg)
        return {'CANCELLED'}

    T, J, _ = posed_joints.shape
    motion_name = os.path.splitext(os.path.basename(filepath))[0]
    
    if J not in [27, 30, 77]:
        msg = f"Addon supports Core (27 joints) and SOMA (30 or 77 joints). Found {J} joints."
        if op:
            op.report({'ERROR'}, msg)
        return {'CANCELLED'}

    # Retrieve global rotations if available
    global_rot_mats = data['global_rot_mats'] if 'global_rot_mats' in data else None

    props = context.scene.ceb_ardy
    scale = props.import_scale
    
    paths, err = get_ardy_paths(context)
    if err:
        if op:
            op.report({'ERROR'}, err)
        return {'CANCELLED'}
        
    arm_obj, rig_joint_names = setup_soma_skin(context, parent_obj=None, J=J, scale=scale, paths=paths, char_name=motion_name)
    if not arm_obj:
        msg = "Failed to build skinned mesh and armature."
        if op:
            op.report({'ERROR'}, msg)
        return {'CANCELLED'}

    # Get skin data NPZ
    skin_path = get_skin_path(paths, J)
    try:
        skin_data = np.load(skin_path)
        bind_rig_transform = skin_data["bind_rig_transform"]
    except Exception as e:
        msg = f"Failed to load skin data: {e}"
        if op:
            op.report({'ERROR'}, msg)
        return {'CANCELLED'}

    s30_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 28, 39, 40, 41, 42, 46, 56, 67, 68, 69, 70, 72, 73, 74, 75]
    s77_to_s30 = {s77_idx: s30_idx for s30_idx, s77_idx in enumerate(s30_indices)}

    # Keyframe bone poses frame by frame
    for t in range(T):
        frame_num = t + 1
        context.scene.frame_set(frame_num)

        joints_t  = posed_joints[t].tolist()
        rot_t     = global_rot_mats[t].tolist() if global_rot_mats is not None else None

        apply_soma_pose(arm_obj, rig_joint_names, bind_rig_transform,
                        joints_t, rot_t, s77_to_s30, J, scale,
                        record_keys=True, frame_num=frame_num)

    context.scene.frame_start = 1
    context.scene.frame_end = T
    context.scene.frame_current = 1

    if op:
        op.report({'INFO'}, f"Successfully imported motion: {motion_name} ({T} frames, {J} joints)")
    return {'FINISHED'}

class CEB_OT_ArdyRunServer(bpy.types.Operator):
    bl_idname = "ceb.ardy_run_server"
    bl_label = "Start Text Encoder Server"
    bl_description = "Start the ARDY text-encoder server in a separate console window to speed up repeated runs"

    def execute(self, context):
        paths, err = get_ardy_paths(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        if not os.path.exists(paths["run_server_script"]):
            self.report({'ERROR'}, f"Could not find server script: {paths['run_server_script']}")
            return {'CANCELLED'}

        try:
            subprocess.Popen(
                [paths["python_exe"], paths["run_server_script"]],
                cwd=paths["ardy_dir"],
                creationflags=0x00000010  # CREATE_NEW_CONSOLE
            )
            self.report({'INFO'}, "Starting Text Encoder Server...")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start server: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

class CEB_OT_ArdyRunDemo(bpy.types.Operator):
    bl_idname = "ceb.ardy_run_demo"
    bl_label = "Run Interactive Demo"
    bl_description = "Launch the interactive real-time humanoid control demo"

    def execute(self, context):
        paths, err = get_ardy_paths(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        if not os.path.exists(paths["run_demo_script"]):
            self.report({'ERROR'}, f"Could not find demo script: {paths['run_demo_script']}")
            return {'CANCELLED'}

        props = context.scene.ceb_ardy
        cmd = [paths["python_exe"], paths["run_demo_script"]]
        if props.quantize_4bit:
            cmd.append("--quantize-4bit")

        try:
            subprocess.Popen(
                cmd,
                cwd=paths["ardy_dir"],
                creationflags=0x00000010  # CREATE_NEW_CONSOLE
            )
            self.report({'INFO'}, f"Launching Viser Web App Demo (4-bit quantization: {props.quantize_4bit})...")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start demo: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

class CEB_OT_ArdyGenerate(bpy.types.Operator):
    bl_idname = "ceb.ardy_generate"
    bl_label = "Generate Motion"
    bl_description = "Run ARDY motion generation in the background"
    
    _timer = None
    process = None
    output_filepath = ""

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self.process is not None:
                status = self.process.poll()
                if status is not None:
                    # Clean up timer
                    context.window_manager.event_timer_remove(self._timer)
                    self._timer = None
                    
                    if status == 0:
                        self.report({'INFO'}, "Motion generation completed successfully!")
                        props = context.scene.ceb_ardy
                        if props.auto_import:
                            import_ardy_npz(self.output_filepath, context, self)
                    else:
                        self.report({'ERROR'}, f"Motion generation failed (Exit code: {status}). Check the popup console.")
                    
                    return {'FINISHED'}
        return {'PASS_THROUGH'}

    def execute(self, context):
        paths, err = get_ardy_paths(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
            
        props = context.scene.ceb_ardy
        
        # Build command-line arguments
        cmd = [
            paths["python_exe"],
            paths["generate_script"],
            props.prompt,
            "--model", props.model,
            "--duration", f"{props.duration}",
            "--diffusion_steps", f"{props.diffusion_steps}",
            "--output", props.output_name
        ]
        
        if props.seed != -1:
            cmd.extend(["--seed", f"{props.seed}"])
            
        self.output_filepath = os.path.join(paths["ardy_dir"], "outputs", f"{props.output_name}.npz")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=paths["ardy_dir"],
                creationflags=0x00000010  # CREATE_NEW_CONSOLE
            )
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start process: {e}")
            return {'CANCELLED'}
            
        # Register a modal timer to poll the process status without hanging Blender
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Generating motion in background...")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.report({'INFO'}, "Motion generation process cancelled.")

from bpy_extras.io_utils import ImportHelper

class CEB_OT_ArdyImportNPZ(bpy.types.Operator, ImportHelper):
    bl_idname = "ceb.ardy_import_npz"
    bl_label = "Import ARDY Motion (.npz)"
    bl_description = "Directly import a generated ARDY motion NPZ file"

    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(default="*.npz", options={'HIDDEN'})

    def execute(self, context):
        return import_ardy_npz(self.filepath, context, self)

class CEB_OT_CleanAnimation(bpy.types.Operator):
    bl_idname = "ceb.clean_animation"
    bl_label = "Clean Animation"
    bl_description = "Clear all keyframes and animation data from the active or selected ARDY character armature"

    def execute(self, context):
        target_arms = []
        
        # 1. Check character selection list first (active character from UI list)
        char = get_active_character(context)
        if char:
            clean_name = char.name.replace(" ", "_")
            arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_name}_Armature"
            arm_obj = bpy.data.objects.get(arm_name)
            if arm_obj:
                target_arms.append(arm_obj)

        # 2. Fallback: Check active object in Viewport (excluding pose constraint ghost armatures)
        if not target_arms:
            active = context.active_object
            if active and active.type == 'ARMATURE' and not active.name.startswith("ARDY_Pose_"):
                target_arms.append(active)
            
        # 3. Fallback: Check selected objects in Viewport
        if not target_arms:
            for obj in context.selected_objects:
                if obj.type == 'ARMATURE' and not obj.name.startswith("ARDY_Pose_") and obj not in target_arms:
                    target_arms.append(obj)
                
        # 4. Fallback: check all characters in character selection list
        if not target_arms:
            props = context.scene.ceb_ardy
            for char in props.characters:
                clean_name = char.name.replace(" ", "_")
                arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_name}_Armature"
                arm_obj = bpy.data.objects.get(arm_name)
                if arm_obj and arm_obj not in target_arms:
                    target_arms.append(arm_obj)

        if not target_arms:
            self.report({'WARNING'}, "No ARDY character armature selected or found for the active character.")
            return {'CANCELLED'}

        for arm_obj in target_arms:
            if arm_obj.animation_data:
                arm_obj.animation_data_clear()
            if arm_obj.pose:
                for b in arm_obj.pose.bones:
                    b.location = (0, 0, 0)
                    b.rotation_quaternion = (1, 0, 0, 0)
                    b.rotation_euler = (0, 0, 0)
                    b.scale = (1, 1, 1)

            # Reset armature object-level position & transform
            arm_obj.location = (0, 0, 0)
            arm_obj.rotation_euler = (0, 0, 0)
            arm_obj.scale = (1, 1, 1)

            # Apply relaxed idle pose (arms down) so the character looks natural after clean
            apply_relaxed_idle_pose(arm_obj)
            
            # Also clear animation data on parent & mesh objects if linked
            if arm_obj.parent and arm_obj.parent.animation_data:
                arm_obj.parent.animation_data_clear()


        # Check active character to clear associated mesh/parent animation data
        char = get_active_character(context)
        if char:
            for o_name in (char.mesh_obj_name, char.parent_obj_name):
                if o_name:
                    o = bpy.data.objects.get(o_name)
                    if o and o.animation_data:
                        o.animation_data_clear()

        context.scene.frame_current = 1
        tag_redraw_view3d(context)

        # Clear queued frames from realtime stream operator if running & reset start_frame to 1
        global _active_stream_operator
        CEB_OT_ArdyRealtimeStream._frame_queue = []
        if _active_stream_operator is not None:
            _active_stream_operator._frame_queue = []
            _active_stream_operator._buffer = ""
            _active_stream_operator._start_frame = 1
            _active_stream_operator._prepared_arm_name = None
            _active_stream_operator._reset_pending = True

        # Send RESET signal to real-time bridge server if connected or reachable
        global _realtime_client, _realtime_running
        sent = False
        if _realtime_client and _realtime_running:
            try:
                _realtime_client.sendall(b"RESET\n")
                if char:
                    active_prompt = get_active_prompt_for_frame(context.scene.ceb_ardy, 1, context=context)
                    send_switch_char_cmd(char, active_prompt, 1, context=context)
                sent = True
            except Exception as e:
                print(f"[CEB Ardy] Failed to send RESET command over socket: {e}")

        if not sent:
            props = context.scene.ceb_ardy
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(("127.0.0.1", props.realtime_port))
                s.sendall(b"RESET\n")
                s.close()
            except Exception:
                pass

        arm_names = ", ".join([o.name for o in target_arms])
        self.report({'INFO'}, f"Cleaned animation data & reset bridge position for: {arm_names}")
        return {'FINISHED'}

class CEB_OT_LoadCharacter(bpy.types.Operator):
    bl_idname = "ceb.load_character"
    bl_label = "Load Character"
    bl_description = "Load (or reload) the ARDY character mesh and armature for the active character into the scene"

    def execute(self, context):
        paths, err = get_ardy_paths(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        try:
            import numpy as np
        except ImportError:
            self.report({'ERROR'}, "NumPy is not available in Blender's Python environment.")
            return {'CANCELLED'}

        char = get_active_character(context)
        if not char:
            self.report({'ERROR'}, "No active character selected.")
            return {'CANCELLED'}

        props = context.scene.ceb_ardy
        scale = props.import_scale

        J = 27 if char.model == 'core' else 77
        char_name = char.name
        clean_prefix = char_name.replace(" ", "_")
        arm_name = f"{clean_prefix}_Armature"
        mesh_name = f"{clean_prefix}_Skin"
        parent_name = f"ARDY_Character_{clean_prefix}"

        for obj_name in (arm_name, mesh_name, parent_name, char.parent_obj_name):
            if obj_name:
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    bpy.data.objects.remove(obj, do_unlink=True)

        arm_obj, rig_joint_names = setup_soma_skin(context, parent_obj=None, J=J, scale=scale, paths=paths, char_name=char_name)
        if not arm_obj:
            self.report({'ERROR'}, f"Failed to build mesh and armature for {char.name}.")
            return {'CANCELLED'}

        # Apply relaxed idle pose so the character looks natural (not T-pose) on load
        apply_relaxed_idle_pose(arm_obj)

        char.arm_obj_name = arm_obj.name
        char.mesh_obj_name = f"{clean_prefix}_Skin"
        char.parent_obj_name = ""

        self.report({'INFO'}, f"Character loaded: {arm_obj.name} ({J} joints, scale={scale})")
        return {'FINISHED'}


class CEB_OT_LoadArdyCore(bpy.types.Operator):
    """Load Ardy Core body armature and skinned mesh (27 joints)"""
    bl_idname = "ceb.load_ardy_core"
    bl_label = "Load Ardy Core"
    bl_description = "Load Ardy Core body armature and skinned mesh (27 joints)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        paths, _ = get_ardy_paths(context)
        skin_path = get_skin_path(paths, J=27)

        if not skin_path or not os.path.exists(skin_path):
            self.report({'ERROR'}, "Could not locate Ardy Core skin NPZ file.")
            return {'CANCELLED'}

        clean_prefix = "Ardy_Core"
        arm_name = f"{clean_prefix}_Armature"
        mesh_name = f"{clean_prefix}_Skin"
        parent_name = f"ARDY_Character_{clean_prefix}"

        for obj_name in (arm_name, mesh_name, parent_name):
            obj = bpy.data.objects.get(obj_name)
            if obj:
                bpy.data.objects.remove(obj, do_unlink=True)

        arm_obj, rig_joint_names = setup_soma_skin(
            context, parent_obj=None, J=27, scale=1.0, paths=paths, char_name=clean_prefix, skin_path=skin_path
        )

        if not arm_obj:
            self.report({'ERROR'}, "Failed to build Ardy Core mesh and armature.")
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        arm_obj.select_set(True)
        context.view_layer.objects.active = arm_obj

        self.report({'INFO'}, f"Ardy Core Body Armature Loaded Successfully: {arm_obj.name}")
        return {'FINISHED'}


class CEB_OT_ArdyStartBridge(bpy.types.Operator):
    bl_idname = "ceb.ardy_start_bridge"
    bl_label = "Start Bridge Process"
    bl_description = "Start the ARDY real-time bridge process in a new console window for the active character"

    def execute(self, context):
        paths, err = get_ardy_paths(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        addon_dir = os.path.dirname(os.path.abspath(__file__))
        bridge_script = os.path.join(addon_dir, "blender_bridge.py")
        if not os.path.exists(bridge_script):
            self.report({'ERROR'}, f"Could not find bridge script in addon folder: {bridge_script}")
            return {'CANCELLED'}

        props = context.scene.ceb_ardy
        char = get_active_character(context)
        model = char.model if char else 'core'

        cmd = [paths["python_exe"], bridge_script, "--port", str(props.realtime_port), "--model", model, "--ardy-dir", paths["ardy_dir"]]
        if props.quantize_4bit:
            cmd.append("--quantize-4bit")

        try:
            subprocess.Popen(
                cmd,
                cwd=paths["ardy_dir"],
                creationflags=0x00000010  # CREATE_NEW_CONSOLE
            )
            self.report({'INFO'}, f"Starting ARDY real-time bridge for {char.name if char else 'active character'} (Model: {model.upper()}, Port: {props.realtime_port})...")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start bridge process: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

def get_stream_action_name(char):
    """
    Generate action name based on character name, set prompts and count of waypoints and pose constraints.
    Example: 'Character_1_walk_W1_C0' or 'Hero_walk, run_W2_C1'
    """
    char_name = char.name if char and char.name else "Character_1"

    if not char:
        return f"{char_name}_ARDY_Stream_W0_C0"

    prompts = [
        item.prompt.strip()
        for item in char.prompt_schedule
        if item.enabled
        and not getattr(item, "has_waypoint", False)
        and not getattr(item, "has_pose_constraint", False)
        and item.prompt
        and item.prompt.strip()
    ]
    if prompts:
        prompt_text = ", ".join(prompts)
    elif char.realtime_prompt and char.realtime_prompt.strip():
        prompt_text = char.realtime_prompt.strip()
    else:
        prompt_text = "walk"

    num_waypoints = sum(1 for item in char.prompt_schedule if item.enabled and getattr(item, "has_waypoint", False))
    num_constraints = sum(1 for item in char.prompt_schedule if item.enabled and getattr(item, "has_pose_constraint", False))

    return f"{char_name}_{prompt_text}_W{num_waypoints}_C{num_constraints}"


def action_has_curves(action):
    """Safely check if an Action has curves or keyframes across all Blender versions (legacy vs slotted/layered)."""
    if not action:
        return False
    try:
        if hasattr(action, "is_empty"):
            return not action.is_empty
    except Exception:
        pass
    try:
        if hasattr(action, "fcurves") and action.fcurves is not None:
            return len(action.fcurves) > 0
    except Exception:
        pass
    try:
        if hasattr(action, "curves") and action.curves is not None:
            return len(action.curves) > 0
    except Exception:
        pass
    try:
        if hasattr(action, "slots") and action.slots is not None:
            return len(action.slots) > 0
    except Exception:
        pass
    return False


def push_action_to_nla_track(arm_obj, action):
    """
    Pushes an action onto a new NLA track for the armature object,
    setting the action and strip as fake user to avoid Blender erasing it.
    """
    if not arm_obj or not action:
        return None

    if not arm_obj.animation_data:
        arm_obj.animation_data_create()

    action.use_fake_user = True

    start_frame = int(action.frame_range[0]) if action_has_curves(action) else 1

    track = arm_obj.animation_data.nla_tracks.new()
    track.name = action.name
    try:
        strip = track.strips.new(name=action.name, start=start_frame, action=action)
        strip.extrapolation = 'HOLD_FORWARD'
        track.mute = False
        print(f"[CEB Ardy] Created NLA track and strip '{action.name}' (fake_user=True, extrapolation=NOTHING)")
        return track
    except Exception as e:
        print(f"[CEB Ardy] Error creating NLA strip for action '{action.name}': {e}")
        return None


def prepare_armature_for_streaming(arm_obj, char, context=None):
    """
    Prepares character armature object for a new streaming session:
    1. Mutes all other NLA layers (tracks) on the character if option is enabled.
    2. Removes / pushes down any active action strip from the current layer.
    3. Creates a new Action named with prompts, waypoints, and constraints count, and sets fake user.
    """
    if not arm_obj:
        return None

    if not arm_obj.animation_data:
        arm_obj.animation_data_create()

    anim_data = arm_obj.animation_data

    if context is None:
        context = bpy.context
    props = getattr(context.scene, "ceb_ardy", None) if hasattr(context, "scene") else None
    should_mute = props.mute_previous_nla_layers if props and hasattr(props, "mute_previous_nla_layers") else True

    # 1. Mute all existing NLA layers (tracks) on the character armature if option is enabled
    if should_mute:
        for track in anim_data.nla_tracks:
            track.mute = True

    # 2. Remove active action strip from main action slot if any (push down first if it has keyframes)
    if anim_data.action:
        old_act = anim_data.action
        if action_has_curves(old_act):
            push_action_to_nla_track(arm_obj, old_act)
        anim_data.action = None

    if arm_obj.pose:
        for b in arm_obj.pose.bones:
            b.location = mathutils.Vector((0.0, 0.0, 0.0))

    # 3. Create a new NLA action for this stream session

    act_name = get_stream_action_name(char)
    new_act = bpy.data.actions.new(name=act_name)
    new_act.use_fake_user = True
    anim_data.action = new_act
    print(f"[CEB Ardy] Prepared new NLA action '{new_act.name}' (fake_user=True, mute_previous={should_mute}) for character '{char.name if char else 'Armature'}'")
    return new_act


MHR_TO_ARDY_BONE_MAP = {
    "root": "Hips",
    "c_spine0": "Spine",
    "c_spine1": "Spine1",
    "c_spine2": "Spine2",
    "c_spine3": "Spine3",
    "c_neck": "Neck",
    "c_head": "Head",
    "r_clavicle": "RightShoulder",
    "r_uparm": "RightArm",
    "r_lowarm": "RightForeArm",
    "r_wrist": "RightHand",
    "l_clavicle": "LeftShoulder",
    "l_uparm": "LeftArm",
    "l_lowarm": "LeftForeArm",
    "l_wrist": "LeftHand",
    "r_upleg": "RightUpLeg",
    "r_lowleg": "RightLeg",
    "r_foot": "RightFoot",
    "r_ball": "RightToeBase",
    "l_upleg": "LeftUpLeg",
    "l_lowleg": "LeftLeg",
    "l_foot": "LeftFoot",
    "l_ball": "LeftToeBase",
}

GEMX_TO_ARDY_BONE_MAP = {
    "Hips": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Chest": "Spine3",
    "Neck1": "Neck",
    "Neck2": "Neck",
    "Head": "Head",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "RightHandThumb1": "RightHandThumb1",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "LeftHandThumb1": "LeftHandThumb1",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
}


def get_bone_map_for_armature(src_arm_obj, tgt_arm_obj):
    """Dynamically determines the bone mapping dictionary based on source armature structure."""
    src_bone_names = set(b.name for b in src_arm_obj.data.bones)
    if "LeftShin" in src_bone_names or "Chest" in src_bone_names or "Root" in src_bone_names:
        print(f"[CEB Ardy] Detected GEMX Armature structure in '{src_arm_obj.name}'.")
        return GEMX_TO_ARDY_BONE_MAP
    elif "l_lowleg" in src_bone_names or "body_world" in src_bone_names:
        print(f"[CEB Ardy] Detected MHR Armature structure in '{src_arm_obj.name}'.")
        return MHR_TO_ARDY_BONE_MAP
    else:
        print(f"[CEB Ardy] Unknown armature structure in '{src_arm_obj.name}'; using direct name match.")
        tgt_bone_names = set(b.name for b in tgt_arm_obj.data.bones)
        return {b: b for b in src_bone_names if b in tgt_bone_names}


def retarget_animation_to_ardy(tgt_arm_obj, src_arm_obj, context=None):
    """
    Retargets animation keyframes from a source armature object (GEMX, MHR, etc.) to the active ARDY Core character armature,
    creating a new Action set as fake user and pushed down to an NLA track.
    """
    if context is None:
        context = bpy.context

    if not src_arm_obj or src_arm_obj.type != 'ARMATURE':
        print("[CEB Ardy] Retarget failed: Source armature object is invalid or not an armature.")
        return None

    if not tgt_arm_obj or tgt_arm_obj.type != 'ARMATURE':
        print("[CEB Ardy] Retarget failed: Target ARDY armature object is invalid or not an armature.")
        return None

    bone_map = get_bone_map_for_armature(src_arm_obj, tgt_arm_obj)
    if not bone_map:
        print("[CEB Ardy] Retarget failed: No compatible bone mapping found.")
        return None

    # Determine frame range from source action or scene
    if src_arm_obj.animation_data and src_arm_obj.animation_data.action:
        src_act = src_arm_obj.animation_data.action
        frame_start = int(src_act.frame_range[0])
        frame_end = int(src_act.frame_range[1])
    else:
        frame_start = context.scene.frame_start
        frame_end = context.scene.frame_end

    char = get_active_character(context)
    char_name = char.name if char else tgt_arm_obj.name.replace("_Armature", "")
    act_name = f"{char_name}_Retarget"

    # Create new action for target armature
    target_act = bpy.data.actions.new(name=act_name)
    target_act.use_fake_user = True

    if not tgt_arm_obj.animation_data:
        tgt_arm_obj.animation_data_create()
    tgt_arm_obj.animation_data.action = target_act

    props = getattr(context.scene, "ceb_ardy", None) if hasattr(context, "scene") else None
    flip_180 = props.retarget_flip_180 if props and hasattr(props, "retarget_flip_180") else False
    R_corr_z = mathutils.Matrix.Rotation(math.pi, 3, 'Z')

    # Build topological bone processing order (parents always before children)
    remaining = list(bone_map.keys())
    src_ordered = []
    while remaining:
        progress = False
        for src_bname in list(remaining):
            src_pbone = src_arm_obj.pose.bones.get(src_bname)
            parent_in_remaining = (src_pbone.parent and src_pbone.parent.name in remaining) if src_pbone else False
            if not parent_in_remaining:
                src_ordered.append(src_bname)
                remaining.remove(src_bname)
                progress = True
        if not progress:
            src_ordered.extend(remaining)
            break

    # Perform frame-by-frame pose retargeting using world-space matrix copy
    for frame in range(frame_start, frame_end + 1):
        context.scene.frame_set(frame)
        context.view_layer.update()

        for src_bone_name in src_ordered:
            tgt_bone_name = bone_map[src_bone_name]
            src_pbone = src_arm_obj.pose.bones.get(src_bone_name)
            tgt_pbone = tgt_arm_obj.pose.bones.get(tgt_bone_name)
            if not src_pbone or not tgt_pbone:
                continue

            tgt_pbone.rotation_mode = 'QUATERNION'
            is_root = (tgt_bone_name == "Hips" or tgt_pbone.parent is None)

            if is_root:
                if flip_180:
                    src_loc = R_corr_z @ src_pbone.location
                    src_rot_arm = (R_corr_z @ src_pbone.matrix.to_3x3()).to_4x4()
                else:
                    src_loc = src_pbone.location.copy()
                    src_rot_arm = src_pbone.matrix.to_3x3().to_4x4()
                tgt_pbone.matrix = mathutils.Matrix.Translation(src_loc) @ src_rot_arm
                tgt_pbone.keyframe_insert(data_path="location", frame=frame)
            else:
                tgt_pbone.location = mathutils.Vector((0.0, 0.0, 0.0))
                if flip_180:
                    tgt_pbone.matrix = R_corr_z.to_4x4() @ src_pbone.matrix
                else:
                    tgt_pbone.matrix = src_pbone.matrix

            tgt_pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            context.view_layer.update()

    # Push to NLA Track
    push_action_to_nla_track(tgt_arm_obj, target_act)
    tgt_arm_obj.animation_data.action = None

    context.scene.frame_start = frame_start
    context.scene.frame_end = frame_end
    context.scene.frame_set(frame_start)
    tag_redraw_view3d(context)

    print(f"[CEB Ardy] Successfully retargeted animation '{src_arm_obj.name}' to '{tgt_arm_obj.name}' ({frame_start}..{frame_end}) as NLA track '{target_act.name}'")
    return target_act


def retarget_mhr_to_ardy(tgt_arm_obj, src_arm_obj, context=None):
    """Backward-compatible wrapper function for retarget_animation_to_ardy."""
    return retarget_animation_to_ardy(tgt_arm_obj, src_arm_obj, context=context)


class CEB_OT_RetargetMHR(bpy.types.Operator):
    bl_idname = "ceb.retarget_mhr"
    bl_label = "Retarget to ARDY Core"
    bl_description = "Retarget animation from source armature object (GEMX_Armature, MHR, etc.) to active ARDY Core character armature"

    source_armature_name: bpy.props.StringProperty(
        name="Source Armature",
        description="Name of the source armature object",
        default="GEMX_Armature"
    )

    def execute(self, context):
        char = get_active_character(context)
        if not char:
            self.report({'ERROR'}, "No active ARDY character found in character list.")
            return {'CANCELLED'}

        clean_prefix = char.name.replace(" ", "_")
        target_arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_prefix}_Armature"
        tgt_arm_obj = bpy.data.objects.get(target_arm_name)

        if not tgt_arm_obj:
            active = context.active_object
            if active and active.type == 'ARMATURE':
                tgt_arm_obj = active

        if not tgt_arm_obj:
            self.report({'ERROR'}, f"Target ARDY character armature '{target_arm_name}' not found. Please Load/Build the character first.")
            return {'CANCELLED'}

        props = getattr(context.scene, "ceb_ardy", None)
        src_name = self.source_armature_name
        if props and hasattr(props, "source_armature_name") and props.source_armature_name:
            src_name = props.source_armature_name

        src_arm_obj = bpy.data.objects.get(src_name)
        if not src_arm_obj:
            # Fallback search for GEMX or MHR armatures in scene
            for obj in context.scene.objects:
                if obj.type == 'ARMATURE' and obj != tgt_arm_obj:
                    if "GEMX" in obj.name.upper() or "MHR" in obj.name.upper():
                        src_arm_obj = obj
                        break

        if not src_arm_obj:
            self.report({'ERROR'}, f"Source armature '{src_name}' not found in scene.")
            return {'CANCELLED'}

        act = retarget_animation_to_ardy(tgt_arm_obj, src_arm_obj, context=context)
        if not act:
            self.report({'ERROR'}, "Failed to retarget animation.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Successfully retargeted '{src_arm_obj.name}' to '{char.name}' as NLA action '{act.name}'.")
        return {'FINISHED'}


def copy_character_prompt_schedule(source_char, target_char, context, offset_vec=(0.0, 0.0, 0.0), offset_waypoints=True):
    if source_char == target_char:
        return
    for item in list(target_char.prompt_schedule):
        remove_waypoint_empty(item)
        remove_pose_constraint_armature(item)
    target_char.prompt_schedule.clear()

    target_char.realtime_prompt = source_char.realtime_prompt

    for src_item in source_char.prompt_schedule:
        new_item = target_char.prompt_schedule.add()
        new_item.prompt = src_item.prompt
        new_item.start_frame = src_item.start_frame
        new_item.enabled = src_item.enabled
        new_item.has_waypoint = src_item.has_waypoint
        if src_item.has_waypoint:
            if offset_waypoints and offset_vec:
                new_item.waypoint_co = (
                    src_item.waypoint_co[0] + offset_vec[0],
                    src_item.waypoint_co[1] + offset_vec[1],
                    src_item.waypoint_co[2] + offset_vec[2]
                )
            else:
                new_item.waypoint_co = src_item.waypoint_co
            get_or_create_waypoint_empty(context, new_item)
        new_item.has_pose_constraint = src_item.has_pose_constraint

    crowd = get_crowd_for_character(target_char.name, context)
    if crowd and crowd.empty_object_name:
        crowd_empty = bpy.data.objects.get(crowd.empty_object_name)
        if crowd_empty:
            parent_character_items_to_crowd(target_char, crowd_empty)
_recorded_crowd_trajectories = {}

def generate_collision_avoidance_waypoints(char_k, char_idx, start_frame, end_frame, avoid_radius, context):
    global _recorded_crowd_trajectories

    props = getattr(context.scene, "ceb_ardy", None) if hasattr(context, "scene") else None

    # 1. Collect dynamic trajectories from previously simulated characters
    prev_trajectories = [
        (name, traj) for name, traj in _recorded_crowd_trajectories.items()
        if name != char_k.name and traj
    ]

    # 2. Collect initial standing locations from unsimulated characters if option is enabled
    unsimulated_positions = []
    if props and getattr(props, "crowd_avoid_unsimulated", True):
        for other_char in props.characters:
            if other_char == char_k:
                continue
            if other_char.name not in _recorded_crowd_trajectories:
                ox, oy, oz, _ = get_character_world_transform(other_char)
                unsimulated_positions.append((other_char.name, mathutils.Vector((ox, oy, oz))))

    if not prev_trajectories and not unsimulated_positions:
        return

    clean_prefix = char_k.name.replace(" ", "_")
    arm_name = char_k.arm_obj_name if char_k.arm_obj_name else f"{clean_prefix}_Armature"
    arm_obj = bpy.data.objects.get(arm_name)
    if not arm_obj:
        return

    start_loc = arm_obj.matrix_world.to_translation().copy()

    dest_loc = start_loc.copy()
    existing_wps = [item for item in char_k.prompt_schedule if item.enabled and getattr(item, "has_waypoint", False)]
    if existing_wps:
        wp_item = existing_wps[0]
        if wp_item.waypoint_object_name and bpy.data.objects.get(wp_item.waypoint_object_name):
            dest_loc = bpy.data.objects[wp_item.waypoint_object_name].matrix_world.to_translation().copy()
        else:
            dest_loc = mathutils.Vector(wp_item.waypoint_co)
    else:
        forward_dir = arm_obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 1.0, 0.0))
        dest_loc = start_loc + forward_dir * 10.0

    total_frames = max(1, end_frame - start_frame)
    step_frames = 20
    min_avoid_frame = start_frame + 15

    # Calculate average forward speed (meters per frame) from recorded trajectories
    speed_m_per_frame = 0.045  # Default ~1.1 m/s at 25 fps
    recorded_speeds = []
    for _name, traj in prev_trajectories:
        if len(traj) >= 2:
            frames = sorted(traj.keys())
            first_f, last_f = frames[0], frames[-1]
            if last_f > first_f:
                total_d = (traj[last_f] - traj[first_f]).length
                recorded_speeds.append(total_d / float(last_f - first_f))
    if recorded_speeds:
        speed_m_per_frame = max(0.02, sum(recorded_speeds) / len(recorded_speeds))

    window_frames = 40  # Check obstacle positions within +-40 frames (~1.6s window)

    for f in range(min_avoid_frame, end_frame + 1, step_frames):
        frames_elapsed = f - start_frame
        estimated_dist = speed_m_per_frame * frames_elapsed

        if existing_wps:
            wp_vec = dest_loc - start_loc
            wp_dist = wp_vec.length
            if wp_dist > 1e-4:
                t_factor = min(1.0, estimated_dist / wp_dist)
                curr_pos = start_loc.lerp(dest_loc, t_factor)
            else:
                curr_pos = start_loc.copy()
        else:
            forward_dir = arm_obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 1.0, 0.0))
            if forward_dir.length > 1e-4:
                forward_dir.normalize()
            curr_pos = start_loc + forward_dir * estimated_dist

        obstacles = []

        # Check dynamic trajectories from previously simulated characters in time window [f - 40, f + 40]
        for traj_name, prev_traj in prev_trajectories:
            close_frames = [pf for pf in prev_traj.keys() if abs(pf - f) <= window_frames]
            for pf in close_frames:
                obstacles.append(prev_traj[pf])

        # Check unsimulated standing obstacles
        for _name, unsim_pos in unsimulated_positions:
            obstacles.append(unsim_pos)

        # Project positions onto the 2D ground plane (X, Y)
        start_ground_z = start_loc.z

        for other_pos in obstacles:
            curr_pos_2d = mathutils.Vector((curr_pos.x, curr_pos.y, start_ground_z))
            other_pos_2d = mathutils.Vector((other_pos.x, other_pos.y, start_ground_z))

            dist = (curr_pos_2d - other_pos_2d).length
            if dist < avoid_radius:
                ray_dir = (dest_loc - start_loc) if existing_wps else (curr_pos - start_loc)
                ray_dir_2d = mathutils.Vector((ray_dir.x, ray_dir.y, 0.0))
                if ray_dir_2d.length > 1e-4:
                    ray_dir_norm = ray_dir_2d.normalized()
                    vec_to_obs = other_pos_2d - curr_pos_2d
                    proj_length = vec_to_obs.dot(ray_dir_norm)
                    proj_vec = ray_dir_norm * proj_length
                    perp_to_obs = vec_to_obs - proj_vec
                    if perp_to_obs.length > 1e-4:
                        away_dir = -perp_to_obs.normalized()
                    else:
                        away_dir = mathutils.Vector((-ray_dir_norm.y, ray_dir_norm.x, 0.0)).normalized()
                else:
                    vec_away = curr_pos_2d - other_pos_2d
                    vec_away.z = 0.0
                    away_dir = vec_away.normalized() if vec_away.length > 1e-4 else mathutils.Vector((1.0, 0.0, 0.0))

                away_dir.z = 0.0
                if away_dir.length > 1e-4:
                    away_dir.normalize()
                else:
                    away_dir = mathutils.Vector((1.0, 0.0, 0.0))

                detour_dist = max(avoid_radius - dist + 0.6, avoid_radius * 0.6)
                detour_offset = away_dir * detour_dist
                detour_co = mathutils.Vector((curr_pos_2d.x + detour_offset.x, curr_pos_2d.y + detour_offset.y, start_ground_z))

                existing_near = any(
                    item.has_waypoint and abs(item.start_frame - f) < step_frames
                    for item in char_k.prompt_schedule
                )
                if not existing_near:
                    item = char_k.prompt_schedule.add()
                    item.start_frame = f
                    item.prompt = "walk"
                    item.enabled = True
                    item.has_waypoint = True
                    item.waypoint_co = (detour_co.x, detour_co.y, detour_co.z)
                    get_or_create_waypoint_empty(context, item)
                    crowd = get_crowd_for_character(char_k.name, context)
                    if crowd and crowd.empty_object_name:
                        crowd_empty = bpy.data.objects.get(crowd.empty_object_name)
                        if crowd_empty:
                            parent_character_items_to_crowd(char_k, crowd_empty)
                    print(f"[CEB Ardy Collision Avoidance] Inserted detour waypoint for '{char_k.name}' at frame {f} (dist={dist:.2f}m < {avoid_radius:.2f}m, detour={detour_dist:.2f}m)")
                    break




class CEB_OT_ClearCrowdSettings(bpy.types.Operator):
    bl_idname = "ceb.clear_crowd_settings"
    bl_label = "Clear Crowd Settings"
    bl_description = "Remove all waypoints and pose constraints for all characters in the selected crowd"

    def execute(self, context):
        props = context.scene.ceb_ardy
        crowd = get_active_crowd(context)
        if not crowd:
            self.report({'ERROR'}, "No active crowd selected.")
            return {'CANCELLED'}

        char_names = [n.strip() for n in crowd.character_names.split(",") if n.strip()]
        cleared_count = 0
        for c_name in char_names:
            char_item = None
            for c in props.characters:
                if c.name == c_name:
                    char_item = c
                    break

            if char_item:
                for pitem in list(char_item.prompt_schedule):
                    remove_waypoint_empty(pitem)
                    remove_pose_constraint_armature(pitem)
                char_item.prompt_schedule.clear()
                item = char_item.prompt_schedule.add()
                item.start_frame = props.crowd_start_frame
                item.prompt = char_item.realtime_prompt if char_item.realtime_prompt else "walk"
                item.enabled = True
                cleared_count += 1

        self.report({'INFO'}, f"Cleared waypoints and constraints for {cleared_count} characters in crowd '{crowd.name}'.")
        return {'FINISHED'}


class CEB_OT_GenerateCrowdAnimation(bpy.types.Operator):

    bl_idname = "ceb.generate_crowd_animation"
    bl_label = "Generate Crowd Animation"
    bl_description = "Generate crowd characters and produce animations for all of them automatically using active character's prompts"

    is_regenerating: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        paths, err = get_ardy_paths(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        props = context.scene.ceb_ardy
        active_crowd = get_active_crowd(context) if self.is_regenerating else None

        start_frame = props.crowd_start_frame
        end_frame = props.crowd_end_frame
        spacing = props.crowd_spacing
        layout_mode = props.crowd_layout
        offset_waypoints = props.crowd_offset_waypoints

        if start_frame >= end_frame:
            self.report({'ERROR'}, "Start frame must be less than end frame.")
            return {'CANCELLED'}

        # Clear recorded trajectories from any previous crowd generation run
        global _recorded_crowd_trajectories
        _recorded_crowd_trajectories.clear()

        # 1. Set scene start and end frames
        context.scene.frame_start = start_frame
        context.scene.frame_end = end_frame
        context.scene.frame_current = start_frame

        # CASE 1: Regenerating with Clear Settings DISABLED
        if self.is_regenerating and active_crowd and not active_crowd.clear_settings_before_generate:
            char_names = [n.strip() for n in active_crowd.character_names.split(",") if n.strip()]
            crowd_char_indices = []
            for c_name in char_names:
                for idx, c in enumerate(props.characters):
                    if c.name == c_name:
                        crowd_char_indices.append(idx)
                        break

            if not crowd_char_indices:
                self.report({'ERROR'}, f"No valid characters found for crowd '{active_crowd.name}'.")
                return {'CANCELLED'}

            reverse_order = getattr(props, "crowd_reverse_order", False)
            char_indices = list(reversed(crowd_char_indices)) if reverse_order else list(crowd_char_indices)
            first_idx = char_indices[0]

            props.active_character_index = first_idx
            tag_redraw_view3d(context)

            global _realtime_running
            if _realtime_running:
                bpy.ops.ceb.ardy_realtime_stream()

            CEB_OT_ArdyRealtimeStream._is_crowd_generating = True
            CEB_OT_ArdyRealtimeStream._crowd_char_indices = char_indices
            CEB_OT_ArdyRealtimeStream._crowd_current_char_idx = 0
            CEB_OT_ArdyRealtimeStream._crowd_start_frame = start_frame
            CEB_OT_ArdyRealtimeStream._crowd_end_frame = end_frame

            try:
                res = bpy.ops.ceb.ardy_realtime_stream()
                if res in ({'FINISHED'}, {'RUNNING_MODAL'}):
                    self.report({'INFO'}, f"Regenerating motion for crowd '{active_crowd.name}' (using existing settings & waypoints)...")
                    return {'FINISHED'}
                else:
                    CEB_OT_ArdyRealtimeStream._is_crowd_generating = False
                    self.report({'ERROR'}, "Failed to start ARDY stream for crowd regeneration.")
                    return {'CANCELLED'}
            except Exception as e:
                CEB_OT_ArdyRealtimeStream._is_crowd_generating = False
                self.report({'ERROR'}, f"Could not connect to ARDY bridge: {e}")
                return {'CANCELLED'}

        # CASE 2: New Crowd OR Regenerating with Clear Settings ENABLED
        source_char = get_active_character(context)
        if not source_char:
            self.report({'ERROR'}, "No active character found in character list.")
            return {'CANCELLED'}

        # If regenerating with Clear Settings enabled, clear waypoints/constraints for crowd's characters
        if self.is_regenerating and active_crowd and active_crowd.clear_settings_before_generate:
            bpy.ops.ceb.clear_crowd_settings()

        target_count = props.crowd_count

        # Ensure target_count characters exist in props.characters
        current_count = len(props.characters)
        if current_count < target_count:
            for idx in range(current_count + 1, target_count + 1):
                name = f"Character_{idx}"
                while any(c.name == name for c in props.characters):
                    idx += 1
                    name = f"Character_{idx}"
                c = props.characters.add()
                c.name = name
                c.model = 'core'

        num_chars = min(target_count, len(props.characters))

        # Calculate spatial positions and headings based on selected layout pattern
        positions = []
        headings = []

        if layout_mode == 'LINE':
            for i in range(num_chars):
                x = (i - (num_chars - 1) / 2.0) * spacing
                y = 0.0
                z = 0.0
                positions.append((x, y, z))
                headings.append(0.0)
        elif layout_mode == 'CIRCLE':
            radius = max(spacing, (num_chars * spacing) / (2 * math.pi))
            for i in range(num_chars):
                angle = (2 * math.pi * i) / num_chars
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = 0.0
                positions.append((x, y, z))
                headings.append(angle)
        elif layout_mode == 'RANDOM':
            import random
            positions = [(0.0, 0.0, 0.0)]
            headings = [0.0]
            max_attempts = 1000
            for i in range(1, num_chars):
                placed = False
                radius_search = spacing * math.sqrt(num_chars)
                for _ in range(max_attempts):
                    rx = random.uniform(-radius_search, radius_search)
                    ry = random.uniform(-radius_search, radius_search)
                    if all(math.hypot(rx - px, ry - py) >= spacing for px, py, _ in positions):
                        positions.append((rx, ry, 0.0))
                        headings.append(random.uniform(0, 2 * math.pi))
                        placed = True
                        break
                if not placed:
                    x = (i % 5 - 2) * spacing
                    y = (i // 5 - 2) * spacing
                    positions.append((x, y, 0.0))
                    headings.append(0.0)
        else:  # GRID default
            cols = math.ceil(math.sqrt(num_chars))
            for i in range(num_chars):
                row_idx = i // cols
                col_idx = i % cols
                x = (col_idx - (cols - 1) / 2.0) * spacing
                y = (row_idx - (math.ceil(num_chars / cols) - 1) / 2.0) * spacing
                z = 0.0
                positions.append((x, y, z))
                headings.append(0.0)

        # Base reference position for Character 0
        x0, y0, z0 = positions[0]

        for i in range(num_chars):
            x, y, z = positions[i]
            heading = headings[i]

            props.active_character_index = i
            char = props.characters[i]

            clean_prefix = char.name.replace(" ", "_")
            arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_prefix}_Armature"
            arm_obj = bpy.data.objects.get(arm_name)

            if not arm_obj:
                bpy.ops.ceb.load_character()
                arm_obj = bpy.data.objects.get(char.arm_obj_name if char.arm_obj_name else f"{clean_prefix}_Armature")

            if arm_obj:
                arm_obj.location = mathutils.Vector((x, y, z))
                arm_obj.rotation_euler.z = heading

            if i != 0 and char != source_char:
                offset_vec = (x - x0, y - y0, z - z0)
                copy_character_prompt_schedule(source_char, char, context, offset_vec=offset_vec, offset_waypoints=offset_waypoints)

        # Parent Empty object setup (reuse if regenerating, else create new)
        col_name = "ARDY_Crowds"
        collection = bpy.data.collections.get(col_name)
        if not collection:
            collection = bpy.data.collections.new(col_name)
            context.scene.collection.children.link(collection)

        avg_x = sum(p[0] for p in positions) / float(num_chars)
        avg_y = sum(p[1] for p in positions) / float(num_chars)

        if self.is_regenerating and active_crowd and active_crowd.empty_object_name and bpy.data.objects.get(active_crowd.empty_object_name):
            crowd_empty = bpy.data.objects[active_crowd.empty_object_name]
            crowd_empty.location = mathutils.Vector((avg_x, avg_y, 0.0))
            crowd_item = active_crowd
        else:
            crowd_idx = len(props.crowds) + 1
            crowd_name = f"Crowd_{crowd_idx}"
            empty_base = f"ARDY_{crowd_name}"
            empty_name = empty_base
            e_cnt = 1
            while bpy.data.objects.get(empty_name):
                empty_name = f"{empty_base}_{e_cnt}"
                e_cnt += 1

            crowd_empty = bpy.data.objects.new(empty_name, None)
            crowd_empty.empty_display_type = 'CUBE'
            crowd_empty.empty_display_size = 1.0
            crowd_empty.location = mathutils.Vector((avg_x, avg_y, 0.0))
            collection.objects.link(crowd_empty)

            crowd_item = props.crowds.add()
            crowd_item.name = crowd_name
            crowd_item.empty_object_name = crowd_empty.name

        crowd_char_names = []
        for i in range(num_chars):
            char = props.characters[i]
            crowd_char_names.append(char.name)
            parent_character_items_to_crowd(char, crowd_empty)

        crowd_item.character_names = ",".join(crowd_char_names)
        crowd_item.crowd_count = num_chars
        crowd_item.layout_mode = layout_mode
        crowd_item.spacing = spacing
        crowd_item.start_frame = start_frame
        crowd_item.end_frame = end_frame
        crowd_item.source_char_name = source_char.name

        # Force Blender dependency graph to evaluate all updated armature locations immediately
        try:
            context.view_layer.update()
        except Exception:
            pass

        # Determine simulation character sequence order
        reverse_order = getattr(props, "crowd_reverse_order", False)
        char_indices = list(reversed(range(num_chars))) if reverse_order else list(range(num_chars))
        first_idx = char_indices[0]

        props.active_character_index = first_idx
        tag_redraw_view3d(context)

        # Generate collision avoidance waypoints for initial character against unsimulated standing characters
        if getattr(props, "crowd_avoid_collisions", True):
            char_first = props.characters[first_idx] if len(props.characters) > first_idx else None
            if char_first:
                generate_collision_avoidance_waypoints(
                    char_first, first_idx, start_frame, end_frame,
                    getattr(props, "crowd_avoid_radius", 1.8), context
                )

        # Start real-time stream in crowd generation mode
        if _realtime_running:
            bpy.ops.ceb.ardy_realtime_stream()

        CEB_OT_ArdyRealtimeStream._is_crowd_generating = True
        CEB_OT_ArdyRealtimeStream._crowd_char_indices = char_indices
        CEB_OT_ArdyRealtimeStream._crowd_current_char_idx = 0
        CEB_OT_ArdyRealtimeStream._crowd_start_frame = start_frame
        CEB_OT_ArdyRealtimeStream._crowd_end_frame = end_frame

        try:
            res = bpy.ops.ceb.ardy_realtime_stream()
            if res in ({'FINISHED'}, {'RUNNING_MODAL'}):
                self.report({'INFO'}, f"Crowd generation started for crowd '{crowd_item.name}' ({num_chars} characters)...")
                return {'FINISHED'}
            else:
                CEB_OT_ArdyRealtimeStream._is_crowd_generating = False
                self.report({'ERROR'}, "Failed to start ARDY stream for crowd generation. Ensure Bridge process is running.")
                return {'CANCELLED'}
        except Exception as e:
            CEB_OT_ArdyRealtimeStream._is_crowd_generating = False
            self.report({'ERROR'}, "Could not connect to ARDY real-time bridge. Please click 'Start Bridge' first.")
            return {'CANCELLED'}




class CEB_OT_ArdyRealtimeStream(bpy.types.Operator):

    bl_idname = "ceb.ardy_realtime_stream"
    bl_label = "Toggle Real-time Stream"
    bl_description = "Connect or disconnect the real-time ARDY motion stream"

    _timer = None
    _buffer = ""
    _frame_queue = None
    _prepared_arm_name = None
    _is_crowd_generating = False
    _crowd_char_indices = []
    _crowd_current_char_idx = 0
    _crowd_start_frame = 1
    _crowd_end_frame = 250

    def modal(self, context, event):

        global _realtime_client, _realtime_running

        if event.type == 'TIMER':
            if _realtime_client is None or not _realtime_running:
                self.cleanup(context)
                return {'FINISHED'}

            import socket
            import json
            
            try:
                data = _realtime_client.recv(8192)
                if data:
                    self._buffer += data.decode("utf-8")
                    while "\n" in self._buffer:
                        line, self._buffer = self._buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                payload = json.loads(line)
                                if self._frame_queue is not None:
                                    self._frame_queue.append(payload)
                            except Exception as parse_err:
                                print(f"Error parsing socket packet: {parse_err}")
                elif len(data) == 0:
                    self.report({'WARNING'}, "Stream closed by bridge server.")
                    self.cleanup(context)
                    return {'FINISHED'}
            except BlockingIOError:
                pass
            except socket.error as e:
                self.report({'ERROR'}, f"Socket error: {e}")
                self.cleanup(context)
                return {'CANCELLED'}

            props = context.scene.ceb_ardy
            current_frame = context.scene.frame_current
            char = get_active_character(context)
            active_prompt = get_active_prompt_for_frame(props, current_frame)
            if char and hasattr(self, "_last_sent_prompt") and self._last_sent_prompt != active_prompt:
                try:
                    char.realtime_prompt = active_prompt
                    prompt_cmd = f"PROMPT:{active_prompt}\n"
                    _realtime_client.sendall(prompt_cmd.encode("utf-8"))
                    self._last_sent_prompt = active_prompt
                    if len(self._frame_queue) > 3:
                        self._frame_queue = self._frame_queue[:3]
                    print(f"[CEB Ardy Stream] Prompt updated at frame {current_frame} → '{active_prompt}'")
                except Exception as pe:
                    print(f"[CEB Ardy Stream] Error sending prompt update: {pe}")

            if self._frame_queue:
                frames_to_process = 1
                if len(self._frame_queue) > 20:
                    frames_to_process = 3
                elif len(self._frame_queue) > 8:
                    frames_to_process = 2

                for _ in range(frames_to_process):
                    if not self._frame_queue:
                        break
                    payload = self._frame_queue.pop(0)
                    joints = payload.get("joints", [])
                    frame_num = payload.get("frame", 0)
                    global_rot_mats = payload.get("global_rot_mats", None)
                    char_name = payload.get("char_name", None)
                    self.update_viewport(context, joints, frame_num, global_rot_mats=global_rot_mats, char_name=char_name, payload=payload)

        return {'PASS_THROUGH'}

    def execute(self, context):
        global _realtime_client, _realtime_running, _active_stream_operator
        props = context.scene.ceb_ardy

        if _realtime_running:
            self.report({'INFO'}, "Disconnecting from ARDY stream...")
            self.cleanup(context)
            return {'FINISHED'}

        import socket
        self.report({'INFO'}, "Connecting to ARDY real-time bridge...")
        
        try:
            _realtime_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _realtime_client.connect(("127.0.0.1", props.realtime_port))
            _realtime_client.setblocking(False)
            _realtime_running = True
            _active_stream_operator = self
            props.realtime_status = "Connected"
            
            self._is_crowd_generating = getattr(CEB_OT_ArdyRealtimeStream, "_is_crowd_generating", False)
            self._crowd_char_indices = getattr(CEB_OT_ArdyRealtimeStream, "_crowd_char_indices", [])
            self._crowd_current_char_idx = getattr(CEB_OT_ArdyRealtimeStream, "_crowd_current_char_idx", 0)
            self._crowd_start_frame = getattr(CEB_OT_ArdyRealtimeStream, "_crowd_start_frame", 1)
            self._crowd_end_frame = getattr(CEB_OT_ArdyRealtimeStream, "_crowd_end_frame", 250)

            CEB_OT_ArdyRealtimeStream._is_crowd_generating = False
            CEB_OT_ArdyRealtimeStream._crowd_char_indices = []
            CEB_OT_ArdyRealtimeStream._crowd_current_char_idx = 0

            self._buffer = ""
            self._frame_queue = []
            self._reset_pending = True

            
            current_frame = context.scene.frame_current
            self._start_frame = current_frame
            char = get_active_character(context)
            active_prompt = get_active_prompt_for_frame(props, current_frame, context=context)
            if char:
                char.realtime_prompt = active_prompt
                clean_prefix = char.name.replace(" ", "_")
                arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_prefix}_Armature"
                arm_obj = bpy.data.objects.get(arm_name)
                if arm_obj:
                    prepare_armature_for_streaming(arm_obj, char, context=context)
                    self._prepared_arm_name = arm_obj.name
                else:
                    self._prepared_arm_name = None
            else:
                self._prepared_arm_name = None

            self._last_sent_prompt = active_prompt

            send_switch_char_cmd(char, active_prompt, current_frame, context=context)
        except Exception as e:
            self.report({'ERROR'}, f"Connection failed: {e}. Is the Bridge Process running?")
            _realtime_client = None
            _realtime_running = False
            _active_stream_operator = None
            self._buffer = ""
            self._frame_queue = None
            self._prepared_arm_name = None
            props.realtime_status = "Disconnected"
            return {'CANCELLED'}

        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        global _realtime_client, _realtime_running, _active_stream_operator
        props = context.scene.ceb_ardy
        
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            
        if _realtime_client is not None:
            try:
                _realtime_client.sendall(b"STOP\n")
                _realtime_client.close()
            except Exception:
                pass
            _realtime_client = None
            
        char = get_active_character(context)
        if char:
            clean_prefix = char.name.replace(" ", "_")
            arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_prefix}_Armature"
            arm_obj = bpy.data.objects.get(arm_name)
            if arm_obj and arm_obj.animation_data and arm_obj.animation_data.action:
                act = arm_obj.animation_data.action
                if action_has_curves(act):
                    push_action_to_nla_track(arm_obj, act)
                arm_obj.animation_data.action = None

        self._buffer = ""
        self._frame_queue = None
        self._prepared_arm_name = None
        self._is_crowd_generating = False
        self._crowd_char_indices = []
        self._crowd_current_char_idx = 0
        _realtime_running = False
        _active_stream_operator = None
        props.realtime_status = "Disconnected"
        self.report({'INFO'}, "ARDY stream disconnected.")

    def update_viewport(self, context, joints, frame_num, global_rot_mats=None, char_name=None, payload=None):
        if not joints:
            return

        char = get_active_character(context)
        if not char:
            return

        # Ignore stale frame packets belonging to a previous character prior to dynamic switch
        if char_name and char.name != char_name:
            return

        # Ignore stale frame packets belonging to a previous reset session / character transform
        if payload and isinstance(payload, dict):
            packet_reset_id = payload.get("reset_id", None)
            active_reset_id = getattr(self, "_current_reset_id", None)
            if packet_reset_id is not None and active_reset_id is not None:
                if packet_reset_id != active_reset_id:
                    return

        props = context.scene.ceb_ardy
        scale = props.import_scale
        J = len(joints)
        
        if J not in [27, 30, 77]:
            return
            
        paths, err = get_ardy_paths(context)
        if err:
            return

        char_name_str = char.name
        clean_prefix = char_name_str.replace(" ", "_")

        arm_name = char.arm_obj_name if (char and char.arm_obj_name) else f"{clean_prefix}_Armature"
        mesh_name = char.mesh_obj_name if (char and char.mesh_obj_name) else f"{clean_prefix}_Skin"
        parent_name = char.parent_obj_name if (char and char.parent_obj_name) else f"ARDY_Character_{clean_prefix}"

        arm_obj = bpy.data.objects.get(arm_name)
        mesh_obj = bpy.data.objects.get(mesh_name)

        if not arm_obj or not mesh_obj or len(arm_obj.pose.bones) == 0:
            for p_name in (parent_name, char.parent_obj_name if char else None):
                if p_name:
                    p_obj = bpy.data.objects.get(p_name)
                    if p_obj:
                        bpy.data.objects.remove(p_obj, do_unlink=True)

            if arm_obj and not mesh_obj:
                bpy.data.objects.remove(arm_obj, do_unlink=True)
            elif mesh_obj and not arm_obj:
                bpy.data.objects.remove(mesh_obj, do_unlink=True)

            arm_obj, rig_joint_names = setup_soma_skin(context, parent_obj=None, J=J, scale=scale, paths=paths, char_name=char_name_str)
            if char:
                char.arm_obj_name = arm_obj.name
                char.mesh_obj_name = f"{clean_prefix}_Skin"
                char.parent_obj_name = ""
        else:
            rig_joint_names = [b.name for b in arm_obj.pose.bones]

        if not arm_obj or not rig_joint_names:
            return

        if getattr(self, "_prepared_arm_name", None) != arm_obj.name:
            prepare_armature_for_streaming(arm_obj, char, context=context)
            self._prepared_arm_name = arm_obj.name

        start_f = getattr(self, "_start_frame", 1)
        if getattr(self, "_reset_pending", False):
            if abs(frame_num - start_f) > 2 and frame_num > start_f:
                print(f"[CEB Ardy Stream] Discarding stale pre-reset frame packet {frame_num} (expected start near frame {start_f})")
                return
            else:
                self._reset_pending = False

        skin_path = get_skin_path(paths, J)
        try:
            if not hasattr(self, "_cached_skin_path") or self._cached_skin_path != skin_path:
                skin_data = np.load(skin_path)
                self._cached_bind_rig_transform = skin_data["bind_rig_transform"]
                self._cached_skin_path = skin_path
            bind_rig_transform = self._cached_bind_rig_transform
        except Exception as e:
            print(f"[CEB Ardy] Failed to load skin transform: {e}")
            return

        s30_indices = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 18, 28, 39, 40, 41, 42, 46, 56, 67, 68, 69, 70, 72, 73, 74, 75]
        s77_to_s30 = {s77_idx: s30_idx for s30_idx, s77_idx in enumerate(s30_indices)}

        context.scene.frame_set(frame_num)

        apply_soma_pose(arm_obj, rig_joint_names, bind_rig_transform,
                        joints, global_rot_mats, s77_to_s30, J, scale,
                        record_keys=props.realtime_recording, frame_num=frame_num)

        if props.realtime_recording:
            context.scene.frame_current += 1

        # Record frame trajectory for crowd collision avoidance
        # Use the actual root joint world position (joints[0] is in ARDY space; convert to Blender world space)
        global _recorded_crowd_trajectories
        if char:
            if char.name not in _recorded_crowd_trajectories:
                _recorded_crowd_trajectories[char.name] = {}
            if joints and len(joints) > 0:
                root_j = joints[0]  # ARDY space: [X_ardy, Y_ardy(up), Z_ardy(forward)]
                # Convert ARDY -> Blender world: X_b=-X_a, Y_b=Z_a, Z_b=Y_a
                root_world_b = mathutils.Vector((
                    -float(root_j[0]) * scale,
                     float(root_j[2]) * scale,
                     float(root_j[1]) * scale,
                ))
                _recorded_crowd_trajectories[char.name][frame_num] = root_world_b
            elif arm_obj:
                # Fallback: armature object location (static, but better than nothing)
                _recorded_crowd_trajectories[char.name][frame_num] = arm_obj.matrix_world.to_translation().copy()

        # Check multi-character crowd generation auto-advance
        if getattr(self, "_is_crowd_generating", False):
            end_f = getattr(self, "_crowd_end_frame", 250)
            if frame_num >= end_f:
                if arm_obj and arm_obj.animation_data and arm_obj.animation_data.action:
                    act = arm_obj.animation_data.action
                    if action_has_curves(act):
                        push_action_to_nla_track(arm_obj, act)
                    arm_obj.animation_data.action = None

                self._crowd_current_char_idx += 1
                if self._crowd_current_char_idx < len(self._crowd_char_indices):
                    next_idx = self._crowd_char_indices[self._crowd_current_char_idx]
                    props.active_character_index = next_idx
                    next_char = props.characters[next_idx]

                    if getattr(props, "crowd_avoid_collisions", True):
                        generate_collision_avoidance_waypoints(
                            next_char, next_idx, self._crowd_start_frame, self._crowd_end_frame,
                            getattr(props, "crowd_avoid_radius", 1.8), context
                        )

                    context.scene.frame_current = self._crowd_start_frame
                    self._start_frame = self._crowd_start_frame

                    clean_p = next_char.name.replace(" ", "_")

                    next_arm_name = next_char.arm_obj_name if next_char.arm_obj_name else f"{clean_p}_Armature"
                    next_arm_obj = bpy.data.objects.get(next_arm_name)
                    if next_arm_obj:
                        prepare_armature_for_streaming(next_arm_obj, next_char, context=context)
                        self._prepared_arm_name = next_arm_obj.name
                    else:
                        self._prepared_arm_name = None

                    self._frame_queue = []
                    self._buffer = ""
                    self._reset_pending = True

                    active_prompt = get_active_prompt_for_frame(props, self._crowd_start_frame, context=context)
                    self._last_sent_prompt = active_prompt

                    send_switch_char_cmd(next_char, active_prompt, self._crowd_start_frame, context=context)
                    print(f"[CEB Ardy Crowd] Switched to character {self._crowd_current_char_idx + 1}/{len(self._crowd_char_indices)} ('{next_char.name}')")
                else:
                    print("[CEB Ardy Crowd] All crowd characters generated successfully!")
                    self._is_crowd_generating = False
                    props.active_character_index = 0
                    context.scene.frame_current = self._crowd_start_frame
                    self.cleanup(context)
                    self.report({'INFO'}, f"Crowd animation complete for {len(self._crowd_char_indices)} characters!")


def copy_evaluated_pose_to_armature(src_arm, dst_arm):
    """
    Copy the evaluated pose from src_arm (which has constraints/IK)
    to dst_arm (the pose constraint or character armature, without keyframes).
    Preserves exact global world space position for root bones.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    src_eval = src_arm.evaluated_get(depsgraph)

    # Sort bones in root-to-leaf (parent-first) order
    ordered_bones = []
    def visit(b):
        if b not in ordered_bones:
            if b.parent and b.parent not in ordered_bones:
                visit(b.parent)
            ordered_bones.append(b)
    for b in dst_arm.data.bones:
        visit(b)

    dst_inv = dst_arm.matrix_world.inverted()

    for b in ordered_bones:
        dst_pb = dst_arm.pose.bones[b.name]
        src_pb = src_eval.pose.bones.get(b.name)
        if not src_pb:
            continue

        dst_pb.rotation_mode = 'QUATERNION'

        if not b.parent:
            # Root bone: use world matrix of src_eval decomposed relative to dst_arm
            src_wmat = src_eval.matrix_world @ src_pb.matrix
            dst_pose_mat = dst_inv @ src_wmat
            delta_mat = b.matrix_local.inverted() @ dst_pose_mat
            dst_pb.location = delta_mat.to_translation()
            dst_pb.rotation_quaternion = delta_mat.to_quaternion()
            dst_pb.scale = delta_mat.to_scale()
        else:
            parent_src_mat = src_pb.parent.matrix
            rel_mat = parent_src_mat.inverted() @ src_pb.matrix
            rel_rest = b.parent.matrix_local.inverted() @ b.matrix_local
            delta_mat = rel_rest.inverted() @ rel_mat
            dst_pb.location = delta_mat.to_translation()
            dst_pb.rotation_quaternion = delta_mat.to_quaternion()
            dst_pb.scale = delta_mat.to_scale()

def align_rig_to_pose_armature(orig_arm, rig_obj):
    """Align loaded Ardy_Core_rig controls to match orig_arm pose in IK mode."""
    # 1. Match rig object world matrix to ground position (Z=0) and heading Z-rotation directly under Hips
    if 'Hips' in orig_arm.pose.bones:
        hips_wmat = orig_arm.matrix_world @ orig_arm.pose.bones['Hips'].matrix
        hips_wpos = hips_wmat.to_translation()
        ground_pos = mathutils.Vector((hips_wpos.x, hips_wpos.y, 0.0))
        char_rot_z = hips_wmat.to_euler().z
    else:
        hips_wpos = orig_arm.matrix_world.to_translation()
        ground_pos = mathutils.Vector((hips_wpos.x, hips_wpos.y, 0.0))
        char_rot_z = orig_arm.matrix_world.to_euler().z

    rig_obj.matrix_world = mathutils.Matrix.Translation(ground_pos) @ mathutils.Matrix.Rotation(char_rot_z, 4, 'Z')
    rig_inv = rig_obj.matrix_world.inverted()

    # Controls mapping: (orig_arm bone, rig control bone, rig qr_offset bone)
    controls_map = [
        ('Hips', 'c_root.x', 'Hips_qr_offset'),
        ('Spine', 'c_spine_01.x', 'Spine_qr_offset'),
        ('Spine1', 'c_spine_02.x', 'Spine1_qr_offset'),
        ('Spine2', 'c_spine_03.x', 'Spine2_qr_offset'),
        ('Spine3', 'c_spine_04.x', 'Spine3_qr_offset'),
        ('Neck', 'c_neck.x', 'Neck_qr_offset'),
        ('Head', 'c_head.x', 'Head_qr_offset'),
        ('RightShoulder', 'c_shoulder.r', 'RightShoulder_qr_offset'),
        ('LeftShoulder', 'c_shoulder.l', 'LeftShoulder_qr_offset'),
        ('RightArm', 'c_arm_fk.r', 'RightArm_qr_offset'),
        ('RightForeArm', 'c_forearm_fk.r', 'RightForeArm_qr_offset'),
        ('RightHand', 'c_hand_fk.r', 'RightHand_qr_offset'),
        ('LeftArm', 'c_arm_fk.l', 'LeftArm_qr_offset'),
        ('LeftForeArm', 'c_forearm_fk.l', 'LeftForeArm_qr_offset'),
        ('LeftHand', 'c_hand_fk.l', 'LeftHand_qr_offset'),
        ('RightUpLeg', 'c_thigh_fk.r', 'RightUpLeg_qr_offset'),
        ('RightLeg', 'c_leg_fk.r', 'RightLeg_qr_offset'),
        ('RightFoot', 'c_foot_fk.r', 'RightFoot_qr_offset'),
        ('LeftUpLeg', 'c_thigh_fk.l', 'LeftUpLeg_qr_offset'),
        ('LeftLeg', 'c_leg_fk.l', 'LeftLeg_qr_offset'),
        ('LeftFoot', 'c_foot_fk.l', 'LeftFoot_qr_offset'),
    ]

    # Set IK mode and pole parenting on pose bones
    for pb in rig_obj.pose.bones:
        if 'ik_fk_switch' in pb:
            pb['ik_fk_switch'] = 0.0
        if 'pole_parent' in pb:
            pb['pole_parent'] = 1
        if 'pole_parenting' in pb:
            pb['pole_parenting'] = 1

    # Calculate control offsets relative to qr_offset rest matrices
    offsets = {}
    for arm_b, ctrl_b, qr_b in controls_map:
        if ctrl_b in rig_obj.pose.bones and qr_b in rig_obj.data.bones:
            qr_rest_wmat = rig_obj.matrix_world @ rig_obj.data.bones[qr_b].matrix_local
            ctrl_rest_wmat = rig_obj.matrix_world @ rig_obj.data.bones[ctrl_b].matrix_local
            offsets[ctrl_b] = qr_rest_wmat.inverted() @ ctrl_rest_wmat

    # Apply target pose from orig_arm to all controls
    for arm_b, ctrl_b, qr_b in controls_map:
        if arm_b in orig_arm.pose.bones and ctrl_b in rig_obj.pose.bones:
            target_wmat = orig_arm.matrix_world @ orig_arm.pose.bones[arm_b].matrix
            if ctrl_b in offsets:
                ctrl_wmat = target_wmat @ offsets[ctrl_b]
            else:
                ctrl_wmat = target_wmat
            rig_obj.pose.bones[ctrl_b].matrix = rig_inv @ ctrl_wmat
            if hasattr(bpy.context, "view_layer") and bpy.context.view_layer:
                bpy.context.view_layer.update()

    # Align IK target controls (hands and feet) taking Child Of constraints and qr_offset rest matrices into account
    ik_targets = [
        ('c_hand_ik.r', 'RightHand', 'RightHand_qr_offset'),
        ('c_hand_ik.l', 'LeftHand', 'LeftHand_qr_offset'),
        ('c_foot_ik.r', 'RightFoot', 'RightFoot_qr_offset'),
        ('c_foot_ik.l', 'LeftFoot', 'LeftFoot_qr_offset'),
    ]
    ik_offsets = {}
    for ik_b, arm_b, qr_b in ik_targets:
        if ik_b in rig_obj.pose.bones and qr_b in rig_obj.data.bones:
            qr_rest_wmat = rig_obj.matrix_world @ rig_obj.data.bones[qr_b].matrix_local
            ik_rest_wmat = rig_obj.matrix_world @ rig_obj.data.bones[ik_b].matrix_local
            ik_offsets[ik_b] = qr_rest_wmat.inverted() @ ik_rest_wmat

    for ik_b, arm_b, qr_b in ik_targets:
        if ik_b in rig_obj.pose.bones and arm_b in orig_arm.pose.bones:
            pb = rig_obj.pose.bones[ik_b]
            orig_wmat = orig_arm.matrix_world @ orig_arm.pose.bones[arm_b].matrix
            if ik_b in ik_offsets:
                target_wmat = orig_wmat @ ik_offsets[ik_b]
            else:
                target_wmat = orig_wmat
                
            childof = next((c for c in pb.constraints if c.type == 'CHILD_OF' and c.influence > 0), None)
            if childof and childof.subtarget in rig_obj.pose.bones:
                sub_wmat = rig_obj.matrix_world @ rig_obj.pose.bones[childof.subtarget].matrix
                child_wmat = sub_wmat @ childof.inverse_matrix
                pb.matrix = child_wmat.inverted() @ target_wmat
            else:
                pb.matrix = rig_inv @ target_wmat
            if hasattr(bpy.context, "view_layer") and bpy.context.view_layer:
                bpy.context.view_layer.update()

    # Set pole targets for IK elbows (behind body) and knees (in front of body) with evaluated verification
    def set_pole(shoulder_name, elbow_name, hand_name, pole_name, is_leg=False):
        if all(b in orig_arm.pose.bones for b in (shoulder_name, elbow_name, hand_name)) and pole_name in rig_obj.pose.bones:
            p_sh = (orig_arm.matrix_world @ orig_arm.pose.bones[shoulder_name].matrix).to_translation()
            p_el = (orig_arm.matrix_world @ orig_arm.pose.bones[elbow_name].matrix).to_translation()
            p_hd = (orig_arm.matrix_world @ orig_arm.pose.bones[hand_name].matrix).to_translation()
            
            if 'c_root_master.x' in rig_obj.pose.bones:
                root_master_wmat = rig_obj.matrix_world @ rig_obj.pose.bones['c_root_master.x'].matrix
                char_fwd = (root_master_wmat.to_quaternion() @ mathutils.Vector((0, 1, 0))).normalized()
            else:
                char_fwd = (orig_arm.matrix_world.to_quaternion() @ mathutils.Vector((0, 1, 0))).normalized()
            
            v_sh_hd = (p_hd - p_sh)
            if v_sh_hd.length > 1e-4:
                v_sh_hd_n = v_sh_hd.normalized()
                proj = p_sh + v_sh_hd_n * (p_el - p_sh).dot(v_sh_hd_n)
                pole_vec = (p_el - proj)
                if pole_vec.length > 1e-4:
                    pole_dir = pole_vec.normalized()
                else:
                    pole_dir = char_fwd if is_leg else -char_fwd
            else:
                pole_dir = char_fwd if is_leg else -char_fwd
                
            if is_leg:
                if pole_dir.dot(char_fwd) < 0:
                    pole_dir = -pole_dir
            else:
                if pole_dir.dot(char_fwd) > 0:
                    pole_dir = -pole_dir
                    
            def assign_pole_matrix(dir_vec):
                pole_pos = p_el + dir_vec * 0.5
                target_wmat = mathutils.Matrix.Translation(pole_pos)
                pb = rig_obj.pose.bones[pole_name]
                childof = next((c for c in pb.constraints if c.type == 'CHILD_OF' and c.influence > 0), None)
                if childof and childof.subtarget in rig_obj.pose.bones:
                    sub_wmat = rig_obj.matrix_world @ rig_obj.pose.bones[childof.subtarget].matrix
                    child_wmat = sub_wmat @ childof.inverse_matrix
                    pb.matrix = child_wmat.inverted() @ target_wmat
                else:
                    pb.matrix = rig_inv @ target_wmat
                if hasattr(bpy.context, "view_layer") and bpy.context.view_layer:
                    bpy.context.view_layer.update()

            assign_pole_matrix(pole_dir)

            # Verification of evaluated world position
            if hasattr(bpy.context, "evaluated_depsgraph_get"):
                dg = bpy.context.evaluated_depsgraph_get()
                dg.update()
                r_eval = rig_obj.evaluated_get(dg)
                eval_wpos = (r_eval.matrix_world @ r_eval.pose.bones[pole_name].matrix).to_translation()
                eval_dir = (eval_wpos - p_el).normalized()
                if is_leg and eval_dir.dot(char_fwd) < 0:
                    assign_pole_matrix(-pole_dir)
                elif not is_leg and eval_dir.dot(char_fwd) > 0:
                    assign_pole_matrix(-pole_dir)

    set_pole('RightArm', 'RightForeArm', 'RightHand', 'c_arms_pole.r', is_leg=False)
    set_pole('LeftArm', 'LeftForeArm', 'LeftHand', 'c_arms_pole.l', is_leg=False)
    set_pole('RightUpLeg', 'RightLeg', 'RightFoot', 'c_leg_pole.r', is_leg=True)
    set_pole('LeftUpLeg', 'LeftLeg', 'LeftFoot', 'c_leg_pole.l', is_leg=True)

def cleanup_ik_control_collection(context):
    """Unlink and delete all objects and collection imported for IK control."""
    props = context.scene.ceb_ardy
    coll_name = props.ik_loaded_collection_name
    if not coll_name:
        coll_name = "Ardy_Core_rig"

    colls_to_remove = [c for c in bpy.data.collections if c.name == coll_name or c.name.startswith(f"{coll_name}.")]
    for col in colls_to_remove:
        for obj in list(col.all_objects):
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception as e:
                print(f"[CEB Ardy] Error removing object '{obj.name}': {e}")
        try:
            bpy.data.collections.remove(col)
        except Exception as e:
            print(f"[CEB Ardy] Error removing collection '{col.name}': {e}")

class CEB_OT_StartIKControl(bpy.types.Operator):
    bl_idname = "ceb.start_ik_control"
    bl_label = "IK Control"
    bl_description = "Load IK Control rig to visually edit pose for character or pose constraint"

    target_type: bpy.props.EnumProperty(
        name="Target Type",
        items=[
            ('CONSTRAINT', "Constraint", "Edit pose constraint"),
            ('CHARACTER', "Character", "Edit character armature pose"),
        ],
        default='CONSTRAINT'
    )

    @classmethod
    def poll(cls, context):
        if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
            return False
        props = context.scene.ceb_ardy
        if props.ik_control_active:
            return False
        char = get_active_character(context)
        if not char:
            return False
        return True

    def execute(self, context):
        props = context.scene.ceb_ardy
        char = get_active_character(context)
        if not char:
            self.report({'ERROR'}, "No active character.")
            return {'CANCELLED'}

        orig_arm = None
        if self.target_type == 'CHARACTER':
            clean_name = char.name.replace(" ", "_")
            arm_name = char.arm_obj_name if char.arm_obj_name else f"{clean_name}_Armature"
            orig_arm = bpy.data.objects.get(arm_name)
            if not orig_arm:
                orig_arm = bpy.data.objects.get("Character_1_Armature") or bpy.data.objects.get(f"{char.name}_Armature")
            if not orig_arm:
                self.report({'ERROR'}, f"Character armature object for '{char.name}' not found.")
                return {'CANCELLED'}
            blend_file_name = "Ardy_Core_Rig_Character.blend"
            coll_to_load = "Andy_Core_Character_rig"
        else:
            if not char.prompt_schedule:
                self.report({'ERROR'}, "No active schedule items.")
                return {'CANCELLED'}
            idx = char.prompt_schedule_index
            if not (0 <= idx < len(char.prompt_schedule)):
                self.report({'ERROR'}, "Invalid schedule item index.")
                return {'CANCELLED'}
            item = char.prompt_schedule[idx]
            if not getattr(item, "has_pose_constraint", False) or not getattr(item, "pose_armature_name", ""):
                self.report({'ERROR'}, "Selected schedule item is not a pose constraint.")
                return {'CANCELLED'}
            orig_arm = bpy.data.objects.get(item.pose_armature_name)
            if not orig_arm:
                self.report({'ERROR'}, f"Pose constraint armature '{item.pose_armature_name}' not found.")
                return {'CANCELLED'}
            blend_file_name = "Ardy_Core_Rig.blend"
            coll_to_load = "Ardy_Core_rig"

        blend_path = os.path.join(os.path.dirname(__file__), blend_file_name)
        if not os.path.exists(blend_path):
            self.report({'ERROR'}, f"Rig file not found: {blend_path}")
            return {'CANCELLED'}

        try:
            with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
                if coll_to_load in data_from.collections:
                    data_to.collections = [coll_to_load]

            if not data_to.collections:
                self.report({'ERROR'}, f"Collection '{coll_to_load}' not found in {blend_file_name}")
                return {'CANCELLED'}

            loaded_coll = data_to.collections[0]
            context.scene.collection.children.link(loaded_coll)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load IK rig collection: {e}")
            return {'CANCELLED'}

        rig_obj = None
        arm_obj = None
        for o in loaded_coll.all_objects:
            if o.type == 'ARMATURE':
                if 'c_root.x' in o.pose.bones or o.name == 'rig' or 'Ardy_Core_rig' in o.name:
                    rig_obj = o
                elif o != orig_arm:
                    arm_obj = o

        if not rig_obj:
            self.report({'ERROR'}, f"Rig object not found in loaded collection '{coll_to_load}'.")
            return {'CANCELLED'}

        # Align rig controls to orig_arm pose
        align_rig_to_pose_armature(orig_arm, rig_obj)

        if arm_obj:
            arm_obj.matrix_world = rig_obj.matrix_world.copy()

        # Collect objects to hide (original armature AND associated mesh/skin objects)
        objects_to_hide = [orig_arm]
        
        # Add any direct children of orig_arm
        for o in bpy.data.objects:
            if o.parent == orig_arm:
                objects_to_hide.append(o)
                
        # Add character mesh/skin and parent objects if target_type == 'CHARACTER'
        if self.target_type == 'CHARACTER' and char:
            clean_name = char.name.replace(" ", "_")
            if char.mesh_obj_name:
                mesh_o = bpy.data.objects.get(char.mesh_obj_name)
                if mesh_o:
                    objects_to_hide.append(mesh_o)
            if char.parent_obj_name:
                parent_o = bpy.data.objects.get(char.parent_obj_name)
                if parent_o:
                    objects_to_hide.append(parent_o)
                    for o in bpy.data.objects:
                        if o.parent == parent_o:
                            objects_to_hide.append(o)
            for o in bpy.data.objects:
                if o.name.startswith(f"{clean_name}_Skin") or o.name.startswith(f"{clean_name}_Mesh") or o.name.startswith(f"{clean_name}_Geo"):
                    objects_to_hide.append(o)

        hidden_names = []
        for obj in objects_to_hide:
            if obj and not obj.hide_get():
                obj.hide_set(True)
                if obj.name not in hidden_names:
                    hidden_names.append(obj.name)

        # Update properties
        props.ik_control_active = True
        props.ik_target_type = self.target_type
        props.ik_loaded_collection_name = loaded_coll.name
        props.ik_original_armature_name = orig_arm.name
        props.ik_hidden_object_names = ",".join(hidden_names)

        # Select rig object and switch to Pose Mode
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.select_all(action='DESELECT')
        rig_obj.select_set(True)
        context.view_layer.objects.active = rig_obj
        bpy.ops.object.mode_set(mode='POSE')

        tag_redraw_view3d(context)
        target_label = "Character" if self.target_type == 'CHARACTER' else "Constraint"
        self.report({'INFO'}, f"IK Control active for {target_label} '{orig_arm.name}'")
        return {'FINISHED'}

class CEB_OT_BakeIKPose(bpy.types.Operator):
    bl_idname = "ceb.bake_ik_pose"
    bl_label = "Bake Pose to Constraint"
    bl_description = "Bake the pose from the IK rig back to the original armature (no keyframes)"

    @classmethod
    def poll(cls, context):
        if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
            return False
        return context.scene.ceb_ardy.ik_control_active

    def execute(self, context):
        props = context.scene.ceb_ardy
        orig_arm_name = props.ik_original_armature_name
        orig_arm = bpy.data.objects.get(orig_arm_name)

        coll_name = props.ik_loaded_collection_name
        coll = bpy.data.collections.get(coll_name)

        rig_obj = None
        arm_obj = None
        if coll:
            for o in coll.all_objects:
                if o.type == 'ARMATURE':
                    if 'c_root.x' in o.pose.bones or o.name == 'rig' or 'Ardy_Core_rig' in o.name:
                        rig_obj = o
                    elif o != orig_arm:
                        arm_obj = o

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        if orig_arm and arm_obj:
            copy_evaluated_pose_to_armature(arm_obj, orig_arm)

        # Unhide original armature AND associated mesh/skin objects
        if orig_arm:
            orig_arm.hide_set(False)

        if props.ik_hidden_object_names:
            for name in props.ik_hidden_object_names.split(","):
                name = name.strip()
                if name:
                    o = bpy.data.objects.get(name)
                    if o:
                        o.hide_set(False)

        if orig_arm:
            bpy.ops.object.select_all(action='DESELECT')
            orig_arm.select_set(True)
            context.view_layer.objects.active = orig_arm

        is_constraint = (getattr(props, "ik_target_type", 'CONSTRAINT') == 'CONSTRAINT')
        cleanup_ik_control_collection(context)

        props.ik_control_active = False
        props.ik_target_type = 'CONSTRAINT'
        props.ik_loaded_collection_name = ""
        props.ik_original_armature_name = ""
        props.ik_hidden_object_names = ""

        tag_redraw_view3d(context)
        if is_constraint:
            send_pose_constraints_to_bridge(context)
            self.report({'INFO'}, "Baked pose to constraint successfully.")
        else:
            self.report({'INFO'}, "Baked pose to character armature successfully.")
        return {'FINISHED'}

class CEB_OT_CancelIKControl(bpy.types.Operator):
    bl_idname = "ceb.cancel_ik_control"
    bl_label = "Cancel IK Control"
    bl_description = "Cancel IK control mode, erasing loaded IK rig and unhiding original armature"

    @classmethod
    def poll(cls, context):
        if not hasattr(context, "scene") or not hasattr(context.scene, "ceb_ardy"):
            return False
        return context.scene.ceb_ardy.ik_control_active

    def execute(self, context):
        props = context.scene.ceb_ardy
        orig_arm_name = props.ik_original_armature_name
        orig_arm = bpy.data.objects.get(orig_arm_name)

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Unhide original armature AND associated mesh/skin objects
        if orig_arm:
            orig_arm.hide_set(False)

        if props.ik_hidden_object_names:
            for name in props.ik_hidden_object_names.split(","):
                name = name.strip()
                if name:
                    o = bpy.data.objects.get(name)
                    if o:
                        o.hide_set(False)

        if orig_arm:
            bpy.ops.object.select_all(action='DESELECT')
            orig_arm.select_set(True)
            context.view_layer.objects.active = orig_arm

        cleanup_ik_control_collection(context)

        props.ik_control_active = False
        props.ik_target_type = 'CONSTRAINT'
        props.ik_loaded_collection_name = ""
        props.ik_original_armature_name = ""
        props.ik_hidden_object_names = ""

        tag_redraw_view3d(context)
        self.report({'INFO'}, "IK Control cancelled.")
        return {'FINISHED'}

class CEB_OT_AddCrowdEntry(bpy.types.Operator):
    bl_idname = "ceb.add_crowd_entry"
    bl_label = "Add Crowd Entry"
    bl_description = "Add a new empty crowd entry"

    def execute(self, context):
        props = context.scene.ceb_ardy
        idx = len(props.crowds) + 1
        item = props.crowds.add()
        item.name = f"Crowd_{idx}"
        props.active_crowd_index = len(props.crowds) - 1
        return {'FINISHED'}

class CEB_OT_RemoveCrowdEntry(bpy.types.Operator):
    bl_idname = "ceb.remove_crowd_entry"
    bl_label = "Remove Crowd Entry"
    bl_description = "Remove selected crowd entry from the list"

    def execute(self, context):
        props = context.scene.ceb_ardy
        if 0 <= props.active_crowd_index < len(props.crowds):
            props.crowds.remove(props.active_crowd_index)
            props.active_crowd_index = max(0, props.active_crowd_index - 1)
        return {'FINISHED'}

class CEB_OT_SelectCrowd(bpy.types.Operator):
    bl_idname = "ceb.select_crowd"
    bl_label = "Select Crowd in Viewport"
    bl_description = "Select the crowd parent Empty and all linked character armatures/meshes in the 3D Viewport"

    def execute(self, context):
        crowd = get_active_crowd(context)
        if not crowd:
            self.report({'ERROR'}, "No active crowd selected.")
            return {'CANCELLED'}

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        for obj in context.view_layer.objects:
            obj.select_set(False)

        selected_count = 0
        empty_obj = bpy.data.objects.get(crowd.empty_object_name) if crowd.empty_object_name else None
        if empty_obj:
            empty_obj.select_set(True)
            context.view_layer.objects.active = empty_obj
            selected_count += 1

        props = context.scene.ceb_ardy
        char_names = [n.strip() for n in crowd.character_names.split(",") if n.strip()]
        for c_name in char_names:
            char_item = None
            if props:
                for c in props.characters:
                    if c.name == c_name:
                        char_item = c
                        break
            clean_prefix = c_name.replace(" ", "_")
            arm_name = char_item.arm_obj_name if (char_item and char_item.arm_obj_name) else f"{clean_prefix}_Armature"
            mesh_name = char_item.mesh_obj_name if (char_item and char_item.mesh_obj_name) else f"{clean_prefix}_Skin"

            for o_name in (arm_name, mesh_name):
                obj = bpy.data.objects.get(o_name)
                if obj:
                    obj.select_set(True)
                    if context.view_layer.objects.active is None:
                        context.view_layer.objects.active = obj
                    selected_count += 1

        self.report({'INFO'}, f"Selected crowd '{crowd.name}' ({selected_count} objects selected).")
        return {'FINISHED'}

class CEB_OT_ToggleHideCrowd(bpy.types.Operator):
    bl_idname = "ceb.toggle_hide_crowd"
    bl_label = "Toggle Hide Crowd"
    bl_description = "Toggle visibility of all objects belonging to this crowd in the 3D Viewport"

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        props = context.scene.ceb_ardy
        if self.index >= 0 and self.index < len(props.crowds):
            crowd = props.crowds[self.index]
        else:
            crowd = get_active_crowd(context)

        if not crowd:
            return {'CANCELLED'}

        crowd.hide_viewport = not crowd.hide_viewport
        update_crowd_hide(crowd, context)
        state_str = "hidden" if crowd.hide_viewport else "visible"
        self.report({'INFO'}, f"Crowd '{crowd.name}' is now {state_str}.")
        return {'FINISHED'}

class CEB_OT_DeleteCrowd(bpy.types.Operator):
    bl_idname = "ceb.delete_crowd"
    bl_label = "Delete Crowd"
    bl_description = "Delete the crowd parent Empty, all linked character armatures, meshes, waypoints, and character entries"

    def execute(self, context):
        props = context.scene.ceb_ardy
        crowd = get_active_crowd(context)
        if not crowd:
            self.report({'ERROR'}, "No active crowd selected.")
            return {'CANCELLED'}

        crowd_name = crowd.name
        empty_obj_name = crowd.empty_object_name
        char_names = [n.strip() for n in crowd.character_names.split(",") if n.strip()]

        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # 1. Remove parent empty object
        if empty_obj_name:
            empty_obj = bpy.data.objects.get(empty_obj_name)
            if empty_obj:
                bpy.data.objects.remove(empty_obj, do_unlink=True)

        # 2. Remove character objects & property entries
        for c_name in char_names:
            char_idx = None
            for idx, c in enumerate(props.characters):
                if c.name == c_name:
                    char_idx = idx
                    break

            if char_idx is not None:
                char_item = props.characters[char_idx]
                for pitem in list(char_item.prompt_schedule):
                    remove_waypoint_empty(pitem)
                    remove_pose_constraint_armature(pitem)

                clean_prefix = char_item.name.replace(" ", "_")
                arm_name = char_item.arm_obj_name if char_item.arm_obj_name else f"{clean_prefix}_Armature"
                mesh_name = char_item.mesh_obj_name if char_item.mesh_obj_name else f"{clean_prefix}_Skin"
                parent_name = char_item.parent_obj_name if char_item.parent_obj_name else f"ARDY_Character_{clean_prefix}"

                for o_name in (arm_name, mesh_name, parent_name):
                    if o_name:
                        obj = bpy.data.objects.get(o_name)
                        if obj:
                            bpy.data.objects.remove(obj, do_unlink=True)

                props.characters.remove(char_idx)

        # 3. Remove crowd item entry
        crowd_idx = props.active_crowd_index
        props.crowds.remove(crowd_idx)
        props.active_crowd_index = max(0, crowd_idx - 1)
        props.active_character_index = max(0, min(props.active_character_index, len(props.characters) - 1))

        self.report({'INFO'}, f"Successfully deleted crowd '{crowd_name}' and all associated characters.")
        return {'FINISHED'}

class CEB_OT_RegenerateCrowd(bpy.types.Operator):
    bl_idname = "ceb.regenerate_crowd"
    bl_label = "Regenerate Crowd"
    bl_description = "Regenerate crowd motion animations for all characters in the selected crowd"

    def execute(self, context):
        props = context.scene.ceb_ardy
        crowd = get_active_crowd(context)
        if not crowd:
            self.report({'ERROR'}, "No active crowd selected.")
            return {'CANCELLED'}

        props.crowd_count = crowd.crowd_count
        props.crowd_layout = crowd.layout_mode
        props.crowd_spacing = crowd.spacing
        props.crowd_start_frame = crowd.start_frame
        props.crowd_end_frame = crowd.end_frame

        return bpy.ops.ceb.generate_crowd_animation(is_regenerating=True)

classes = (
    CEB_Ardy_PromptItem,
    CEB_Ardy_Character,
    CEB_Ardy_CrowdItem,
    CEB_Ardy_SceneProperties,
    CEB_OT_AddCharacterEntry,
    CEB_OT_RemoveCharacterEntry,
    CEB_OT_AddCrowdEntry,
    CEB_OT_RemoveCrowdEntry,
    CEB_OT_SelectCrowd,
    CEB_OT_ToggleHideCrowd,
    CEB_OT_DeleteCrowd,
    CEB_OT_ClearCrowdSettings,
    CEB_OT_RegenerateCrowd,
    CEB_OT_AddPromptItem,
    CEB_OT_AddWaypoint,
    CEB_OT_RemoveWaypoint,
    CEB_OT_AddPoseConstraint,
    CEB_OT_CapturePoseConstraint,
    CEB_OT_RemovePoseConstraint,
    CEB_OT_SelectPromptItem,
    CEB_OT_RemovePromptItem,
    CEB_OT_ClearPromptItems,
    CEB_OT_MovePromptItem,
    CEB_OT_SortPromptItems,
    CEB_OT_LoadCharacter,
    CEB_OT_LoadArdyCore,
    CEB_OT_ArdyRunServer,
    CEB_OT_ArdyRunDemo,
    CEB_OT_ArdyImportNPZ,
    CEB_OT_CleanAnimation,
    CEB_OT_ArdyStartBridge,
    CEB_OT_ArdyRealtimeStream,
    CEB_OT_GenerateCrowdAnimation,
    CEB_OT_RetargetMHR,

    CEB_OT_StartIKControl,
    CEB_OT_BakeIKPose,
    CEB_OT_CancelIKControl,
)

def register():
    global _overlay_draw_handler, _3d_draw_handler
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ceb_ardy = bpy.props.PointerProperty(type=CEB_Ardy_SceneProperties)
    
    if _overlay_draw_handler is None:
        _overlay_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_prompt_overlay_px, (None, None), 'WINDOW', 'POST_PIXEL'
        )
    if _3d_draw_handler is None:
        _3d_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_waypoints_3d_view, (None, None), 'WINDOW', 'POST_VIEW'
        )
        
    if ardy_depsgraph_sync_waypoints not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(ardy_depsgraph_sync_waypoints)

def unregister():
    global _overlay_draw_handler, _3d_draw_handler
    if _overlay_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_overlay_draw_handler, 'WINDOW')
        _overlay_draw_handler = None
    if _3d_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_3d_draw_handler, 'WINDOW')
        _3d_draw_handler = None
        
    if ardy_depsgraph_sync_waypoints in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(ardy_depsgraph_sync_waypoints)

    del bpy.types.Scene.ceb_ardy
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
