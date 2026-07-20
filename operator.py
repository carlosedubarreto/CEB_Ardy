import bpy
import os
import sys
import subprocess
import math
import mathutils
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent

try:
    import numpy as np
except ImportError:
    pass

_realtime_client = None
_realtime_running = False
_overlay_draw_handler = None

def tag_redraw_view3d(context=None):
    if context is None:
        context = bpy.context
    if hasattr(context, "window_manager") and context.window_manager:
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

def update_prompt_item(self, context):
    tag_redraw_view3d(context)

def update_overlay_visibility(self, context):
    tag_redraw_view3d(context)

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
        default=1,
        min=0,
        update=update_prompt_item
    )
    enabled: bpy.props.BoolProperty(
        name="Enabled",
        description="Whether this prompt entry is enabled",
        default=True,
        update=update_prompt_item
    )

def update_realtime_prompt(self, context):
    global _realtime_client
    if _realtime_client:
        try:
            prompt_cmd = f"PROMPT:{self.realtime_prompt}\n"
            _realtime_client.sendall(prompt_cmd.encode("utf-8"))
        except Exception as e:
            print(f"[CEB Ardy] Failed to send prompt over socket: {e}")

class CEB_Ardy_SceneProperties(bpy.types.PropertyGroup):
    model: bpy.props.EnumProperty(
        name="Model",
        description="Model to use for motion",
        items=[
            ('core', "CORE", "CORE 27-joint skeleton"),
            ('soma', "SOMA", "SOMA 77-joint skeleton")
        ],
        default='core'
    )
    import_scale: bpy.props.FloatProperty(
        name="Import Scale",
        description="Scale factor applied to joint coordinates",
        default=1.0,
        min=0.001
    )
    realtime_prompt: bpy.props.StringProperty(
        name="Live Prompt",
        description="Text prompt sent to ARDY in real-time",
        default="walk",
        update=update_realtime_prompt
    )
    realtime_recording: bpy.props.BoolProperty(
        name="Live Record",
        description="Record the incoming real-time motion stream as keyframes",
        default=False
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
    prompt_schedule: bpy.props.CollectionProperty(
        type=CEB_Ardy_PromptItem
    )
    prompt_schedule_index: bpy.props.IntProperty(
        name="Active Prompt Index",
        default=0
    )
    show_prompt_overlay: bpy.props.BoolProperty(
        name="Show Overlay in 3D View",
        description="Display real-time prompt overlay in 3D Viewport",
        default=True,
        update=update_overlay_visibility
    )

class CEB_OT_AddPromptItem(bpy.types.Operator):
    bl_idname = "ceb.add_prompt_item"
    bl_label = "Add Prompt"
    bl_description = "Add a new prompt schedule item"

    def execute(self, context):
        props = context.scene.ceb_ardy
        item = props.prompt_schedule.add()
        item.start_frame = context.scene.frame_current
        item.prompt = "walk"
        props.prompt_schedule_index = len(props.prompt_schedule) - 1
        tag_redraw_view3d(context)
        return {'FINISHED'}

class CEB_OT_RemovePromptItem(bpy.types.Operator):
    bl_idname = "ceb.remove_prompt_item"
    bl_label = "Remove Prompt"
    bl_description = "Remove the selected prompt schedule item"

    def execute(self, context):
        props = context.scene.ceb_ardy
        idx = props.prompt_schedule_index
        if 0 <= idx < len(props.prompt_schedule):
            props.prompt_schedule.remove(idx)
            props.prompt_schedule_index = max(0, idx - 1)
            tag_redraw_view3d(context)
        return {'FINISHED'}

class CEB_OT_MovePromptItem(bpy.types.Operator):
    bl_idname = "ceb.move_prompt_item"
    bl_label = "Move Prompt"
    bl_description = "Move selected prompt item up or down"

    direction: bpy.props.EnumProperty(
        items=[('UP', 'Up', ''), ('DOWN', 'Down', '')]
    )

    def execute(self, context):
        props = context.scene.ceb_ardy
        idx = props.prompt_schedule_index
        schedule = props.prompt_schedule
        if self.direction == 'UP' and idx > 0:
            schedule.move(idx, idx - 1)
            props.prompt_schedule_index -= 1
        elif self.direction == 'DOWN' and idx < len(schedule) - 1:
            schedule.move(idx, idx + 1)
            props.prompt_schedule_index += 1
        tag_redraw_view3d(context)
        return {'FINISHED'}

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

    current_frame = context.scene.frame_current
    scene_start = context.scene.frame_start
    scene_end = context.scene.frame_end
    
    sorted_schedule = sorted(props.prompt_schedule, key=lambda x: x.start_frame)
    enabled_schedule = [item for item in sorted_schedule if item.enabled]

    if not enabled_schedule and not props.realtime_prompt:
        return

    active_item = None
    active_prompt_text = props.realtime_prompt if props.realtime_prompt else "None"
    
    for item in sorted_schedule:
        if item.enabled and current_frame >= item.start_frame:
            active_item = item
            active_prompt_text = item.prompt

    max_sched_frame = max([item.start_frame for item in sorted_schedule], default=scene_end)
    frame_min = scene_start
    frame_max = max(scene_end, max_sched_frame + 20)
    total_frames = max(1, frame_max - frame_min)

    font_id = 0

    card_w = min(680, max(380, int(region.width * 0.65)))
    card_h = 108

    x = int((region.width - card_w) / 2)
    y = 35

    try:
        gpu.state.blend_set('ALPHA')
    except Exception:
        pass

    # Background card
    draw_round_rect_2d(x, y, card_w, card_h, (0.08, 0.10, 0.15, 0.88))
    # Top accent line
    draw_round_rect_2d(x, y + card_h - 4, card_w, 4, (0.15, 0.65, 0.95, 0.9))

    # Header text
    try:
        blf.size(font_id, 10)
        blf.color(font_id, 0.55, 0.65, 0.75, 1.0)
        blf.position(font_id, x + 16, y + card_h - 22, 0)
        blf.draw(font_id, "PROMPT TIMELINE")
    except Exception:
        pass

    active_str = f"► Active: \"{active_prompt_text}\" (Frame {current_frame})"
    try:
        blf.size(font_id, 12)
        blf.color(font_id, 0.2, 0.95, 0.45, 1.0)
        blf.position(font_id, x + 140, y + card_h - 22, 0)
        blf.draw(font_id, active_str)
    except Exception:
        pass

    # Timeline track
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

    # Prompt blocks & markers
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

    # Playhead needle
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
    
    sorted_schedule = sorted([item for item in props.prompt_schedule if item.enabled], key=lambda x: x.start_frame)
    active_prompt = None
    for item in sorted_schedule:
        if current_frame >= item.start_frame:
            active_prompt = item.prompt
            
    if active_prompt and active_prompt != props.realtime_prompt:
        props.realtime_prompt = active_prompt

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

def ardy_pos_to_blender(pos_ardy, scale=1.0):
    """Convert ARDY position (X-Right, Y-Up, Z-Forward) to Blender position (X-Right, Y-Forward, Z-Up)."""
    return mathutils.Vector((float(pos_ardy[0]) * scale, float(pos_ardy[2]) * scale, float(pos_ardy[1]) * scale))

def ardy_rot_to_blender(R_a):
    """Convert ARDY 3x3 rotation matrix to Blender 3x3 rotation matrix by swapping Y and Z axes."""
    m = mathutils.Matrix.Identity(3)
    if R_a is not None:
        m[0][0] = float(R_a[0][0]); m[0][1] = float(R_a[0][2]); m[0][2] = float(R_a[0][1])
        m[1][0] = float(R_a[2][0]); m[1][1] = float(R_a[2][2]); m[1][2] = float(R_a[2][1])
        m[2][0] = float(R_a[1][0]); m[2][1] = float(R_a[1][2]); m[2][2] = float(R_a[1][1])
    return m

def ardy_transform_to_blender_matrix(pos_ardy, R_a, scale=1.0):
    """Construct a 4x4 Blender matrix from ARDY position and 3x3 rotation."""
    t_b = ardy_pos_to_blender(pos_ardy, scale)
    R_b = ardy_rot_to_blender(R_a)
    mat = R_b.to_4x4()
    mat.translation = t_b
    return mat

def get_skin_path(paths, J):
    if J == 27:
        rel = os.path.join("ardy", "assets", "skeletons", "cskel27", "skin_standard.npz")
    else:
        rel = os.path.join("ardy", "assets", "skeletons", "somaskel77", "skin_standard.npz")
    return os.path.join(paths["ardy_dir"], rel)

def setup_soma_skin(context, parent_obj, J, scale, paths):
    try:
        import numpy as np
    except ImportError:
        return None, None
        
    skin_path = get_skin_path(paths, J)
    if not os.path.exists(skin_path):
        print(f"[CEB Ardy] Skin path not found: {skin_path}")
        return None, None
        
    try:
        skin_data = np.load(skin_path)
    except Exception as e:
        print(f"[CEB Ardy] Failed to load skin npz: {e}")
        return None, None
        
    bind_vertices = skin_data["bind_vertices"]
    faces = skin_data["faces"]
    bind_rig_transform = skin_data["bind_rig_transform"]
    rig_joint_names = [str(n) for n in skin_data["rig_joint_names"]]
    lbs_indices = skin_data["lbs_indices"]
    lbs_weights = skin_data["lbs_weights"]
    rig_joint_connections = skin_data["rig_joint_connections"]
    
    parent_obj.rotation_euler = (0, 0, 0)

    prefix = "Core" if J == 27 else "SOMA"

    # 1. Create Mesh
    mesh_data = bpy.data.meshes.new(name=f"{prefix}_Skin_Mesh")
    mesh_obj = bpy.data.objects.new(f"{prefix}_Skin", mesh_data)
    context.scene.collection.objects.link(mesh_obj)
    
    verts = [tuple(ardy_pos_to_blender(v, scale)) for v in bind_vertices]
    faces_list = [tuple(f) for f in faces]
    mesh_data.from_pydata(verts, [], faces_list)
    mesh_data.update()
    
    mesh_obj.parent = parent_obj
    
    # 2. Create Armature
    arm_data = bpy.data.armatures.new(f"{prefix}_Armature_Data")
    arm_obj = bpy.data.objects.new(f"{prefix}_Armature", arm_data)
    context.scene.collection.objects.link(arm_obj)
    arm_obj.parent = parent_obj
    
    # Add Armature Modifier
    arm_mod = mesh_obj.modifiers.new(name=f"{prefix}_Armature_Mod", type='ARMATURE')
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
            target_matrix = posed_mat_b @ bind_mat_b.inverted() @ bone.bone.matrix_local
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

    parent_obj = bpy.data.objects.new(f"ARDY_{motion_name}", None)
    context.scene.collection.objects.link(parent_obj)
    parent_obj.rotation_euler = (0, 0, 0)
    
    props = context.scene.ceb_ardy
    scale = props.import_scale
    
    paths, err = get_ardy_paths(context)
    if err:
        if op:
            op.report({'ERROR'}, err)
        return {'CANCELLED'}
        
    arm_obj, rig_joint_names = setup_soma_skin(context, parent_obj, J, scale, paths)
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

        try:
            subprocess.Popen(
                [paths["python_exe"], paths["run_demo_script"]],
                cwd=paths["ardy_dir"],
                creationflags=0x00000010  # CREATE_NEW_CONSOLE
            )
            self.report({'INFO'}, "Launching Interactive Demo...")
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

class CEB_OT_ArdyStartBridge(bpy.types.Operator):
    bl_idname = "ceb.ardy_start_bridge"
    bl_label = "Start Bridge Process"
    bl_description = "Start the ARDY real-time bridge process in a new console window"

    def execute(self, context):
        paths, err = get_ardy_paths(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        bridge_script = os.path.join(paths["ardy_dir"], "scripts", "blender_bridge.py")
        if not os.path.exists(bridge_script):
            self.report({'ERROR'}, f"Could not find bridge script: {bridge_script}")
            return {'CANCELLED'}

        props = context.scene.ceb_ardy
        try:
            subprocess.Popen(
                [paths["python_exe"], bridge_script, "--port", str(props.realtime_port), "--model", props.model],
                cwd=paths["ardy_dir"],
                creationflags=0x00000010  # CREATE_NEW_CONSOLE
            )
            self.report({'INFO'}, "Starting ARDY real-time bridge...")
        except Exception as e:
            self.report({'ERROR'}, f"Failed to start bridge process: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}

class CEB_OT_ArdyRealtimeStream(bpy.types.Operator):
    bl_idname = "ceb.ardy_realtime_stream"
    bl_label = "Toggle Real-time Stream"
    bl_description = "Connect or disconnect the real-time ARDY motion stream"

    _timer = None
    _buffer = ""
    _frame_queue = None

    def modal(self, context, event):
        global _realtime_client, _realtime_running

        if event.type == 'TIMER':
            if _realtime_client is None or not _realtime_running:
                self.cleanup(context)
                return {'FINISHED'}

            import socket
            import json
            
            # 1. Read socket data into buffer and parse complete JSON frame packets into queue
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

            # 2. Process frames from the unimported motion data buffer
            if self._frame_queue:
                # Dynamic catch-up if buffer builds up significantly (> 20 or > 40 frames)
                frames_to_process = 1
                if len(self._frame_queue) > 40:
                    frames_to_process = 3
                elif len(self._frame_queue) > 20:
                    frames_to_process = 2

                for _ in range(frames_to_process):
                    if not self._frame_queue:
                        break
                    payload = self._frame_queue.pop(0)
                    joints = payload.get("joints", [])
                    frame_num = payload.get("frame", 0)
                    global_rot_mats = payload.get("global_rot_mats", None)
                    self.update_viewport(context, joints, frame_num, global_rot_mats=global_rot_mats)

        return {'PASS_THROUGH'}

    def execute(self, context):
        global _realtime_client, _realtime_running
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
            props.realtime_status = "Connected"
            
            self._buffer = ""
            self._frame_queue = []
            
            # Send initial model and prompt
            initial_cmd = f"MODEL:{props.model}\nPROMPT:{props.realtime_prompt}\n"
            _realtime_client.sendall(initial_cmd.encode("utf-8"))
        except Exception as e:
            self.report({'ERROR'}, f"Connection failed: {e}. Is the Bridge Process running?")
            _realtime_client = None
            _realtime_running = False
            self._buffer = ""
            self._frame_queue = None
            props.realtime_status = "Disconnected"
            return {'CANCELLED'}

        self._timer = context.window_manager.event_timer_add(0.05, window=context.window)  # 20 FPS matching ARDY
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        global _realtime_client, _realtime_running
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
            
        self._buffer = ""
        self._frame_queue = None
        _realtime_running = False
        props.realtime_status = "Disconnected"
        self.report({'INFO'}, "ARDY stream disconnected.")

    def update_viewport(self, context, joints, frame_num, global_rot_mats=None):
        if not joints:
            return
            
        props = context.scene.ceb_ardy
        scale = props.import_scale
        J = len(joints)
        
        if J not in [27, 30, 77]:
            return
            
        paths, err = get_ardy_paths(context)
        if err:
            return

        prefix = "Core" if J == 27 else "SOMA"
        arm_name = f"{prefix}_Armature"
        mesh_name = f"{prefix}_Skin"
        parent_name = f"ARDY_Realtime_Skeleton_{prefix}"

        # 1. Find existing ARDY character armature & skin mesh
        arm_obj = bpy.data.objects.get(arm_name)
        mesh_obj = bpy.data.objects.get(mesh_name)

        # 2. If armature or skin mesh is missing, construct character via setup_soma_skin
        if not arm_obj or not mesh_obj or len(arm_obj.pose.bones) == 0:
            parent_obj = bpy.data.objects.get(parent_name)
            if not parent_obj:
                parent_obj = bpy.data.objects.new(parent_name, None)
                context.scene.collection.objects.link(parent_obj)
                parent_obj.rotation_euler = (0, 0, 0)

            # Clean up partial object if one exists without the other
            if arm_obj and not mesh_obj:
                bpy.data.objects.remove(arm_obj, do_unlink=True)
            elif mesh_obj and not arm_obj:
                bpy.data.objects.remove(mesh_obj, do_unlink=True)

            arm_obj, rig_joint_names = setup_soma_skin(context, parent_obj, J, scale, paths)
        else:
            rig_joint_names = [b.name for b in arm_obj.pose.bones]

        if not arm_obj or not rig_joint_names:
            return

        # 3. Load skin transform data (with caching)
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

        # Set frame
        context.scene.frame_set(frame_num)

        apply_soma_pose(arm_obj, rig_joint_names, bind_rig_transform,
                        joints, global_rot_mats, s77_to_s30, J, scale,
                        record_keys=props.realtime_recording, frame_num=frame_num)

        if props.realtime_recording:
            context.scene.frame_current += 1

classes = (
    CEB_Ardy_PromptItem,
    CEB_Ardy_SceneProperties,
    CEB_OT_AddPromptItem,
    CEB_OT_RemovePromptItem,
    CEB_OT_MovePromptItem,
    CEB_OT_ArdyRunServer,
    CEB_OT_ArdyRunDemo,
    CEB_OT_ArdyImportNPZ,
    CEB_OT_ArdyStartBridge,
    CEB_OT_ArdyRealtimeStream,
)

def register():
    global _overlay_draw_handler
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ceb_ardy = bpy.props.PointerProperty(type=CEB_Ardy_SceneProperties)
    
    if _overlay_draw_handler is None:
        _overlay_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_prompt_overlay_px, (None, None), 'WINDOW', 'POST_PIXEL'
        )
        
    if ardy_frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(ardy_frame_change_handler)

def unregister():
    global _overlay_draw_handler
    if _overlay_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_overlay_draw_handler, 'WINDOW')
        _overlay_draw_handler = None
        
    if ardy_frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(ardy_frame_change_handler)

    del bpy.types.Scene.ceb_ardy
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
