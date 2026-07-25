"""Blender ARDY Real-Time Bridge

Streams real ARDY-generated motion (posed_joints + global_rot_mats) over a TCP socket.
Uses the same generation pipeline as run_demo.py (ModelLoadingMixin + GenerationMixin).
"""

import sys
import os
import time
import json
import socket
import errno
import argparse
import threading
import numpy as np

# ─── ARDY Path Setup ─────────────────────────────────────────────────────────
# Parse --ardy-dir or locate ARDY root directory prior to importing ARDY modules
parser_path = argparse.ArgumentParser(add_help=False)
parser_path.add_argument("--ardy-dir", type=str, default=None, help="Path to ARDY root folder")
temp_args, _ = parser_path.parse_known_args()

ardy_root = None
if temp_args.ardy_dir and os.path.isdir(temp_args.ardy_dir):
    ardy_root = os.path.abspath(temp_args.ardy_dir)
elif os.environ.get("ARDY_DIR") and os.path.isdir(os.environ["ARDY_DIR"]):
    ardy_root = os.path.abspath(os.environ["ARDY_DIR"])
else:
    # Fallback search relative to working directory or script location
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "scripts", "interactive_demo")) or os.path.isdir(os.path.join(cwd, "ardy")):
        ardy_root = cwd
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "Portable_Ardy", "ardy"),
            os.path.join(script_dir, "..", "ardy"),
            os.path.join(script_dir, "ardy"),
        ]
        for candidate in candidates:
            cand_abs = os.path.abspath(candidate)
            if os.path.isdir(cand_abs) and os.path.isdir(os.path.join(cand_abs, "scripts", "interactive_demo")):
                ardy_root = cand_abs
                break

if ardy_root:
    ardy_root = os.path.abspath(ardy_root)
    if ardy_root not in sys.path:
        sys.path.insert(0, ardy_root)
    scripts_dir = os.path.join(ardy_root, "scripts")
    if os.path.isdir(scripts_dir) and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        os.chdir(ardy_root)
    except Exception as e:
        print(f"[Bridge] Warning: Could not chdir to {ardy_root}: {e}")
    print(f"[Bridge] ARDY root configured: {ardy_root}")
else:
    print("[Bridge] Warning: ARDY root directory could not be automatically resolved.")

# Also add this script's directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ─── ARDY imports ─────────────────────────────────────────────────────────────
HAS_ARDY = False
try:
    import torch
    from interactive_demo.common import *  # noqa: F401,F403
    from interactive_demo.loading import ModelLoadingMixin
    from interactive_demo.generation import GenerationMixin
    from interactive_demo.gen_constraints import GenConstraintsMixin
    from interactive_demo.constraints import ConstraintsMixin
    from interactive_demo.embedding_cache import CachedTextEncoder
    HAS_ARDY = True
    print("[Bridge] ARDY imports succeeded – real generation available.")
except Exception as e:
    print(f"[Bridge] ARDY import failed: {e}")
    print("[Bridge] ERROR: ARDY model dependencies unavailable. Procedural fallback has been removed.")


# ─── Real ARDY generation bridge session ──────────────────────────────────────
class DummyGUIElement:
    def __init__(self, value=None):
        self.value = value
        self.disabled = False
        self.label = ""

class DummyGE:
    def __init__(self):
        self.gui_replan_buffer_size = DummyGUIElement(8)
        self.gui_history_crop_length = DummyGUIElement(128)
        self.gui_num_samples = DummyGUIElement(1)
        self.gui_diffusion_steps_slider = DummyGUIElement(10)
        self.gui_cfg_text_weight = DummyGUIElement(2.5)
        self.gui_cfg_constraint_weight = DummyGUIElement(2.5)
        self.gui_enable_postprocess_checkbox = DummyGUIElement(False)
        self.gui_future_crop_length = DummyGUIElement(128)
        self.gui_frame_idx_input = DummyGUIElement(0)
        self.gui_replan_trigger_thresh = DummyGUIElement(16)
        self.gui_enable_auto_replan_checkbox = DummyGUIElement(False)
        self.gui_actual_fps = DummyGUIElement(0.0)
        self.gui_current_time = DummyGUIElement(0.0)
        self.gui_compile_mode = DummyGUIElement("None")

    def __getattr__(self, name):
        elem = DummyGUIElement()
        setattr(self, name, elem)
        return elem

class DummyViserScene:
    def add_label(self, *args, **kwargs):
        return DummyGUIElement()
    def add_mesh_simple(self, *args, **kwargs):
        return DummyGUIElement()
    def add_line_segments(self, *args, **kwargs):
        return DummyGUIElement()
    def remove_by_name(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: DummyGUIElement()

class DummyViserServer:
    def __init__(self):
        self.scene = DummyViserScene()

class MinimalFullBodyKeyframeSet:
    """Headless replacement for FullbodyKeyframeSet.

    Stores full-body pose keyframes (joints_pos, joints_rot) without any Viser
    visualization so it works correctly in the headless bridge process.
    The interface exposed here matches what GenConstraintsMixin.compute_model_constraints_lst
    expects (get_constraint_info / get_frame_idx / clear).
    """

    def __init__(self, name: str = "Full-Body", skeleton=None, **kwargs):
        self.name = name
        self.display_name = name
        self.skeleton = skeleton
        # frame_idx -> {"joints_pos": np.ndarray [J,3], "joints_rot": np.ndarray [J,3,3]}
        self.keyframes = {}

    def add_keyframe(self, keyframe_id: str, frame_idx: int,
                     joints_pos, joints_rot,
                     viz_label: bool = False, exists_ok: bool = False, **kwargs):
        if joints_pos is None or joints_rot is None:
            return
        # Convert to numpy for storage
        if hasattr(joints_pos, "cpu"):
            joints_pos = joints_pos.detach().cpu().numpy()
        if hasattr(joints_rot, "cpu"):
            joints_rot = joints_rot.detach().cpu().numpy()
        joints_pos = np.array(joints_pos, dtype=np.float32)
        joints_rot = np.array(joints_rot, dtype=np.float32)
        self.keyframes[frame_idx] = {
            "joints_pos": joints_pos,
            "joints_rot": joints_rot,
        }
        print(f"[Bridge] MinimalFullBodyKeyframeSet: stored pose constraint at frame {frame_idx} ({joints_pos.shape[0]} joints)")

    def get_frame_idx(self):
        return sorted(self.keyframes.keys())

    def get_constraint_info(self, device=None):
        if not self.keyframes:
            return {"frame_idx": [], "joints_pos": None, "joints_rot": None}

        frame_indices = []
        all_joints_pos = []
        all_joints_rot = []
        for fidx in sorted(self.keyframes.keys()):
            frame_indices.append(fidx)
            all_joints_pos.append(self.keyframes[fidx]["joints_pos"])
            all_joints_rot.append(self.keyframes[fidx]["joints_rot"])

        pos_np = np.stack(all_joints_pos, axis=0)   # [N, J, 3]
        rot_np = np.stack(all_joints_rot, axis=0)   # [N, J, 3, 3]

        import torch as _torch
        pos_t = _torch.from_numpy(pos_np)
        rot_t = _torch.from_numpy(rot_np)
        if device:
            pos_t = pos_t.to(device)
            rot_t = rot_t.to(device)

        return {
            "frame_idx": frame_indices,
            "joints_pos": pos_t,
            "joints_rot": rot_t,
        }

    def clear(self, frame_idx=None):
        if frame_idx is None:
            self.keyframes.clear()
        elif frame_idx in self.keyframes:
            del self.keyframes[frame_idx]

class _MinimalClientSession:
    """Minimal session object matching what GenerationMixin expects."""
    def __init__(self):
        self.client = None
        self.model = None
        self.motion_rep = None
        self.motion_tensor = None
        self.joints_pos = None
        self.joints_rot = None
        self.foot_contacts = None
        self.root_velocities = None
        self.frame_idx = 0
        self.max_frame_idx = -1
        self.playing = False
        self.stop_playback = False
        self.text_embedding = None
        self.init_global_translation = np.zeros(3, dtype=np.float32)
        self.init_first_heading_angle = 0.0
        self.gen_horizon_len = 32
        self.num_frames_per_token = 4
        self.max_window_len = 128
        self.motion_tensor_lock = threading.Lock()
        self.replan_lock = threading.Lock()
        self.constraints = {}
        if HAS_ARDY:
            try:
                from ardy.viz.viser_utils import RootKeyframe2DSet
                self.constraints["2D Root"] = RootKeyframe2DSet(
                    name="2D Root",
                    server=DummyViserServer(),
                    skeleton=None,
                )
            except Exception as e:
                print(f"[Bridge] Could not initialize RootKeyframe2DSet: {e}")
        # MinimalFullBodyKeyframeSet is always available (no Viser needed)
        self.constraints["Full-Body"] = MinimalFullBodyKeyframeSet(name="Full-Body", skeleton=None)
        self.timeline_data = None
        self.model_fps = 20.0
        self.ref_character = None
        self.ref_joints_pos = None
        self.ref_joints_rot = None
        self.mujoco_converter = None
        self.mesh_mode = "soma_skin"

        self.gui_elements = DummyGE()


if HAS_ARDY:
    class ArdyBridgeGenerator(ModelLoadingMixin, GenerationMixin, GenConstraintsMixin, ConstraintsMixin):
        """Uses ARDY's real model pipeline identical to run_demo.py."""

        def _build_text_encoder(self):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Bridge] Loading local text encoder (LLM2Vec) on {device} (4-bit quantization={self.quantize_4bit})...")
            return load_text_encoder(mode="local", device=device, quantize_4bit=self.quantize_4bit)

        def __init__(self, model_name="core", quantize_4bit=False):
            self.quantize_4bit = quantize_4bit
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"[Bridge] Using device: {self.device}")
            self.text_encoder = CachedTextEncoder(self._build_text_encoder())
            self.client_sessions = {}
            # Fake client object (no Viser needed)
            self._client_id = 0
            self.client_sessions[self._client_id] = _MinimalClientSession()
            self.max_keyframe_num = 6
            print(f"[Bridge] Loading ARDY model '{model_name}'...")
            self.load_model(self._client_id, model_name)
            session = self.client_sessions[self._client_id]
            session.model_fps = getattr(session.model, "fps", 20.0)
            if session.motion_rep is not None:
                if "2D Root" in session.constraints and session.constraints["2D Root"] is not None:
                    session.constraints["2D Root"].skeleton = session.motion_rep.skeleton
                if "Full-Body" in session.constraints and session.constraints["Full-Body"] is not None:
                    session.constraints["Full-Body"].skeleton = session.motion_rep.skeleton
            print(f"[Bridge] Model loaded. FPS={session.model_fps}")

        def client_active(self, client_id):
            return client_id in self.client_sessions

        def add_character(self, client_id, skeleton, sample_idx):
            pass  # No Viser scene needed

        def set_text(self, prompt: str):
            session = self.client_sessions[self._client_id]
            device = self.device
            tensor, _ = self.text_encoder([prompt])
            session.text_embedding = tensor.to(device)

        def _generate_step(self, client_id: int):
            super()._generate_step(client_id)
            session = self.client_sessions[client_id]
            if hasattr(session, "motion_rep") and session.motion_rep is not None:
                from ardy.skeleton import SOMASkeleton30
                if isinstance(session.motion_rep.skeleton, SOMASkeleton30) and session.joints_pos is not None:
                    if session.joints_pos.shape[2] == 30:
                        with session.motion_tensor_lock:
                            skel30 = session.motion_rep.skeleton
                            skel77 = skel30.somaskel77
                            local_rot_30 = skel30.global_rots_to_local_rots(session.joints_rot)
                            local_rot_77 = skel30.to_SOMASkeleton77(local_rot_30)
                            root_pos = session.joints_pos[:, :, skel30.root_idx]
                            
                            B, T = local_rot_77.shape[:2]
                            local_rot_77_flat = local_rot_77.reshape(B * T, 77, 3, 3)
                            root_pos_flat = root_pos.reshape(B * T, 3)
                            
                            global_rot_77_flat, posed_pos_77_flat, _ = skel77.fk(local_rot_77_flat, root_pos_flat)
                            session.joints_pos = posed_pos_77_flat.reshape(B, T, 77, 3)
                            session.joints_rot = global_rot_77_flat.reshape(B, T, 77, 3, 3)

        def _async_generate(self, client_id: int):
            session = self.client_sessions[client_id]
            with session.replan_lock:
                self._generate_step(client_id)

        def get_next_frame(self):
            """Returns (joints_pos, joints_rot) numpy arrays for the current frame, triggering
            autoregressive generation as needed asynchronously."""
            session = self.client_sessions[self._client_id]
            frame_idx = session.frame_idx

            # Initial synchronous load if no frames exist yet
            if session.joints_pos is None:
                with session.replan_lock:
                    self._generate_step(self._client_id)
            # Background pre-fetch when within 16 frames of window end
            elif session.max_frame_idx - frame_idx <= 16:
                if not session.replan_lock.locked():
                    threading.Thread(
                        target=self._async_generate,
                        args=(self._client_id,),
                        daemon=True
                    ).start()

            if session.joints_pos is None:
                return None, None

            with session.motion_tensor_lock:
                fi = min(frame_idx, session.joints_pos.shape[1] - 1)
                # joints_pos: [B, T, J, 3] → [J, 3]
                pos = session.joints_pos[0, fi].cpu().numpy()
                # joints_rot: [B, T, J, 3, 3] → [J, 3, 3]
                rot = session.joints_rot[0, fi].cpu().numpy()

            session.frame_idx += 1
            return pos, rot
else:
    ArdyBridgeGenerator = None


def reset_generator_session(generator, current_prompt=None):
    if generator is None:
        return
    session = generator.client_sessions.get(generator._client_id)
    if session is None:
        return

    with session.replan_lock:
        with session.motion_tensor_lock:
            session.frame_idx = 0
            session.init_global_translation = np.zeros(3, dtype=np.float32)
            session.init_first_heading_angle = 0.0
            session.joints_pos = None
            session.joints_rot = None
            session.motion_tensor = None
            session.foot_contacts = None
            session.root_velocities = None
            session.max_frame_idx = -1
            if "2D Root" in session.constraints and session.constraints["2D Root"] is not None:
                try:
                    session.constraints["2D Root"].clear()
                except Exception as ce:
                    print(f"[Bridge] Error clearing waypoints: {ce}")
            if "Full-Body" in session.constraints and session.constraints["Full-Body"] is not None:
                try:
                    session.constraints["Full-Body"].clear()
                except Exception as ce:
                    print(f"[Bridge] Error clearing pose constraints: {ce}")

    if current_prompt and hasattr(generator, "set_text"):
        try:
            generator.set_text(current_prompt)
        except Exception as e:
            print(f"[Bridge] Error setting text on reset: {e}")


# ─── Socket server ────────────────────────────────────────────────────────────
def run_server(port, model_name="core", quantize_4bit=False):
    generator = None
    if HAS_ARDY:
        try:
            generator = ArdyBridgeGenerator(model_name=model_name, quantize_4bit=quantize_4bit)
            generator.set_text("walk")
        except Exception as e:
            print(f"[Bridge] Real generator failed to initialize: {e}")
            print("[Bridge] ERROR: Generator initialization failed. Motion generation disabled.")
            generator = None
    else:
        print("[Bridge] ERROR: ARDY imports are not available. Motion generation disabled.")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind(("127.0.0.1", port))
        server.listen(1)
        print(f"[Bridge Server] Listening on localhost:{port}")
    except Exception as e:
        print(f"[Bridge Server] Failed to bind to port {port}: {e}")
        return

    current_prompt = "walk"
    frame_num = 0
    accum_offset_x = 0.0
    accum_offset_z = 0.0
    last_sent_root = None
    prompt_just_changed = False

    while True:
        print("[Bridge Server] Waiting for Blender connection...")
        try:
            client, addr = server.accept()
            print(f"[Bridge Server] Connected: {addr}")
            client.setblocking(True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Bridge Server] Accept error: {e}")
            break

        running = True
        buf = ""
        current_char_name = None
        frame_interval = 1.0 / 20.0  # 20 FPS matching ARDY model rate
        next_frame_time = time.time()

        while running:
            # 1. Read commands from Blender (non-blocking via select)
            import select
            readable, _, _ = select.select([client], [], [], 0)
            if readable:
                try:
                    data = client.recv(65536)
                    if data:
                        buf += data.decode("utf-8")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if line.startswith("SWITCH_CHAR:"):
                                # Syntax: SWITCH_CHAR:char_name:model:prompt:frame:x:y:z:heading
                                try:
                                    parts = line.split(":")
                                    if len(parts) >= 9:
                                        char_name = parts[1]
                                        current_char_name = char_name
                                        new_model = parts[2]
                                        new_prompt = parts[3]
                                        blender_frame = int(parts[4])
                                        char_x = float(parts[5])
                                        char_y = float(parts[6])
                                        char_z = float(parts[7])
                                        char_heading = float(parts[8])

                                        print(f"[Bridge] SWITCH_CHAR: '{char_name}' (model={new_model}) at Blender pos=({char_x:.2f}, {char_y:.2f}, {char_z:.2f}), heading={char_heading:.2f}, prompt='{new_prompt}'")
                                        
                                        current_prompt = new_prompt
                                        frame_num = blender_frame
                                        
                                        if generator is not None:
                                            session = generator.client_sessions[generator._client_id]
                                            
                                            # 1. Switch the model if it changed dynamically
                                            loaded_model = getattr(session, "model_name", None)
                                            if loaded_model != new_model:
                                                print(f"[Bridge] Switching model from '{loaded_model}' to '{new_model}'...")
                                                with session.replan_lock:
                                                    with session.motion_tensor_lock:
                                                        generator.load_model(generator._client_id, new_model)
                                                        session.model_name = new_model
                                                        if session.motion_rep is not None:
                                                            if "2D Root" in session.constraints and session.constraints["2D Root"] is not None:
                                                                session.constraints["2D Root"].skeleton = session.motion_rep.skeleton
                                                            if "Full-Body" in session.constraints and session.constraints["Full-Body"] is not None:
                                                                session.constraints["Full-Body"].skeleton = session.motion_rep.skeleton
                                            
                                            # 2. Reset the simulation session for the new character
                                            reset_generator_session(generator, current_prompt)
                                            
                                            # 3. Setup the initial position and rotation in ARDY space
                                            # Blender (X, Y, Z) -> ARDY (X, 0.0, Y)
                                            session.init_global_translation = np.array([char_x, 0.0, char_y], dtype=np.float32)
                                            session.init_first_heading_angle = char_heading
                                            
                                            # 4. Setup starting offset variables for correct shifting
                                            accum_offset_x = 0.0
                                            accum_offset_z = 0.0
                                            last_sent_root = [char_x, char_z, char_y]
                                            prompt_just_changed = False
                                except Exception as e:
                                    print(f"[Bridge] Error handling SWITCH_CHAR command: {e}")
                            elif line.startswith("PROMPT:"):
                                new_prompt = line[len("PROMPT:"):]
                                current_prompt = new_prompt
                                print(f"[Bridge] Prompt → '{current_prompt}'")
                                if generator is not None:
                                    generator.set_text(current_prompt)
                                    session = generator.client_sessions[generator._client_id]
                                    if hasattr(generator, "restart_from_now") and session.motion_tensor is not None and session.frame_idx > 0:
                                        print(f"[Bridge] Conditioning '{current_prompt}' on current character pose & location at frame {session.frame_idx}")
                                        generator.restart_from_now(generator._client_id)
                                    else:
                                        reset_generator_session(generator, current_prompt)
                                        prompt_just_changed = True
                            elif line.startswith("MODEL:"):
                                new_model = line[len("MODEL:"):].strip()
                                if generator is not None:
                                    session = generator.client_sessions[generator._client_id]
                                    loaded_model = getattr(session, "model_name", None)
                                    if loaded_model != new_model:
                                        print(f"[Bridge] Switching model from '{loaded_model}' to '{new_model}'...")
                                        try:
                                            with session.replan_lock:
                                                with session.motion_tensor_lock:
                                                    generator.load_model(generator._client_id, new_model)
                                                    session.model_name = new_model
                                                    if session.motion_rep is not None:
                                                        if "2D Root" in session.constraints and session.constraints["2D Root"] is not None:
                                                            session.constraints["2D Root"].skeleton = session.motion_rep.skeleton
                                                        if "Full-Body" in session.constraints and session.constraints["Full-Body"] is not None:
                                                            session.constraints["Full-Body"].skeleton = session.motion_rep.skeleton
                                                    reset_generator_session(generator, current_prompt)
                                        except Exception as me:
                                            print(f"[Bridge] Error switching model dynamically: {me}")
                            elif line == "CLEAR_WAYPOINTS":
                                print("[Bridge] Clearing 2D Root waypoints.")
                                if generator is not None:
                                    session = generator.client_sessions[generator._client_id]
                                    if "2D Root" in session.constraints and session.constraints["2D Root"] is not None:
                                        try:
                                            session.constraints["2D Root"].clear()
                                        except Exception as ce:
                                            print(f"[Bridge] Error clearing waypoints: {ce}")
                            elif line.startswith("WAYPOINT:"):
                                try:
                                    parts = line.split(":")
                                    if len(parts) >= 5:
                                        wp_frame = int(parts[1])
                                        wp_xb = float(parts[2])
                                        wp_yb = float(parts[3])
                                        wp_zb = float(parts[4])

                                        if generator is not None:
                                            session = generator.client_sessions[generator._client_id]
                                            if "2D Root" not in session.constraints or session.constraints["2D Root"] is None:
                                                if HAS_ARDY:
                                                    from ardy.viz.viser_utils import RootKeyframe2DSet
                                                    skel = session.motion_rep.skeleton if session.motion_rep else None
                                                    session.constraints["2D Root"] = RootKeyframe2DSet(
                                                        name="2D Root", server=DummyViserServer(), skeleton=skel
                                                    )

                                            root_constraint = session.constraints.get("2D Root")
                                            if root_constraint is not None:
                                                import torch
                                                # Convert Blender world coordinates (wp_xb, wp_yb, wp_zb) -> ARDY model space
                                                wp_x_ardy = -wp_xb - accum_offset_x
                                                wp_z_ardy = wp_yb - accum_offset_z
                                                root_pos_ardy = torch.tensor([wp_x_ardy, 0.0, wp_z_ardy], dtype=torch.float32)
                                                wp_id = f"waypoint_{wp_frame}"
                                                root_constraint.add_keyframe(
                                                    keyframe_id=wp_id,
                                                    frame_idx=wp_frame,
                                                    root_pos=root_pos_ardy,
                                                    viz_label=False,
                                                    exists_ok=True,
                                                    update_path=False,
                                                    add_annulus=False
                                                )
                                                print(f"[Bridge] Received 2D Root Waypoint: Frame {wp_frame} at Blender ({wp_xb:.2f}, {wp_yb:.2f}, {wp_zb:.2f}) -> ARDY Model Space ({wp_x_ardy:.2f}, 0.0, {wp_z_ardy:.2f})")

                                                if hasattr(generator, "restart_from_now") and session.motion_tensor is not None and session.frame_idx > 0:
                                                    generator.restart_from_now(generator._client_id)
                                                elif session.motion_tensor is not None and session.frame_idx == 0:
                                                    reset_generator_session(generator, current_prompt)
                                                    prompt_just_changed = True
                                except Exception as wpe:
                                    print(f"[Bridge] Error processing WAYPOINT command: {wpe}")
                            elif line == "CLEAR_POSE_CONSTRAINTS":
                                print("[Bridge] Clearing Full-Body pose constraints.")
                                if generator is not None:
                                    session = generator.client_sessions[generator._client_id]
                                    if "Full-Body" in session.constraints and session.constraints["Full-Body"] is not None:
                                        try:
                                            session.constraints["Full-Body"].clear()
                                        except Exception as ce:
                                            print(f"[Bridge] Error clearing pose constraints: {ce}")
                            elif line.startswith("POSE_CONSTRAINT:"):
                                try:
                                    parts = line.split(":", 3)
                                    if len(parts) >= 4:
                                        p_frame = int(parts[1])
                                        pos_data = json.loads(parts[2])
                                        rot_data = json.loads(parts[3])

                                        if generator is not None:
                                            session = generator.client_sessions[generator._client_id]
                                            if "Full-Body" not in session.constraints or session.constraints["Full-Body"] is None:
                                                skel = session.motion_rep.skeleton if session.motion_rep else None
                                                session.constraints["Full-Body"] = MinimalFullBodyKeyframeSet(
                                                    name="Full-Body", skeleton=skel
                                                )

                                            fb_constraint = session.constraints.get("Full-Body")
                                            if fb_constraint is not None:
                                                pos_np = np.array(pos_data, dtype=np.float32)
                                                rot_np = np.array(rot_data, dtype=np.float32)

                                                # Subtract accum_offset from root joint position (joint 0)
                                                pos_np[0, 0] -= accum_offset_x
                                                pos_np[0, 2] -= accum_offset_z

                                                wp_id = f"pose_{p_frame}"
                                                fb_constraint.add_keyframe(
                                                    keyframe_id=wp_id,
                                                    frame_idx=p_frame,
                                                    joints_pos=pos_np,
                                                    joints_rot=rot_np,
                                                    viz_label=False,
                                                    exists_ok=True
                                                )

                                                if hasattr(generator, "restart_from_now") and session.motion_tensor is not None and session.frame_idx > 0:
                                                    generator.restart_from_now(generator._client_id)
                                                elif session.motion_tensor is not None and session.frame_idx == 0:
                                                    reset_generator_session(generator, current_prompt)
                                                    prompt_just_changed = True
                                except Exception as pce:
                                    print(f"[Bridge] Error processing POSE_CONSTRAINT command: {pce}")
                            elif line.startswith("FRAME:"):
                                try:
                                    blender_frame = int(line[len("FRAME:"):])
                                    if generator is not None:
                                        session = generator.client_sessions[generator._client_id]
                                        frame_num = blender_frame
                                        if blender_frame == 0:
                                            print(f"[Bridge] Frame 0 start: resetting generator session for fresh motion.")
                                            accum_offset_x = 0.0
                                            accum_offset_z = 0.0
                                            last_sent_root = None
                                            reset_generator_session(generator, current_prompt)
                                            prompt_just_changed = False
                                        elif session.motion_tensor is not None and session.joints_pos is not None:
                                            max_len = session.joints_pos.shape[1]
                                            if 0 <= blender_frame < max_len:
                                                session.frame_idx = blender_frame
                                            else:
                                                session.frame_idx = max_len - 1
                                                if hasattr(generator, "restart_from_now"):
                                                    print(f"[Bridge] Re-conditioning generation from frame {session.frame_idx}")
                                                    generator.restart_from_now(generator._client_id)
                                        else:
                                            # Fresh generation start: start internal generator tensor index at 0
                                            session.frame_idx = 0
                                        print(f"[Bridge] Frame sync: Blender frame={blender_frame}, Generator internal frame_idx={session.frame_idx}")
                                except Exception as fe:
                                    print(f"[Bridge] Frame sync error: {fe}")
                            elif line in ("RESET", "CLEAN"):
                                print("[Bridge] RESET received: Clearing motion history and position offsets.")
                                accum_offset_x = 0.0
                                accum_offset_z = 0.0
                                last_sent_root = None
                                prompt_just_changed = False
                                frame_num = 1
                                reset_generator_session(generator, current_prompt)
                            elif line == "STOP":
                                print("[Bridge] STOP received.")
                                running = False
                                break
                    elif len(data) == 0:
                        print("[Bridge] Connection closed by Blender.")
                        running = False
                        break
                except socket.error as e:
                    print(f"[Bridge] Recv error: {e}")
                    running = False
                    break

            if not running:
                break

            # 2. Get frame data
            joints = None
            rot_list = None
            if generator is not None:
                try:
                    pos, rot = generator.get_next_frame()
                    if pos is not None:
                        if prompt_just_changed:
                            if last_sent_root is not None:
                                accum_offset_x += (last_sent_root[0] - float(pos[0, 0]))
                                accum_offset_z += (last_sent_root[2] - float(pos[0, 2]))
                            prompt_just_changed = False

                        pos_shifted = pos.copy()
                        pos_shifted[:, 0] += accum_offset_x
                        pos_shifted[:, 2] += accum_offset_z

                        last_sent_root = [float(pos_shifted[0, 0]), float(pos_shifted[0, 1]), float(pos_shifted[0, 2])]

                        joints = pos_shifted.tolist()
                        rot_list = rot.tolist()  # [J, 3, 3]
                    else:
                        if frame_num % 100 == 0:
                            print(f"[Bridge] Waiting for generator frame... (frame {frame_num})")
                except Exception as e:
                    print(f"[Bridge] Generator error: {e}")
            else:
                if frame_num % 100 == 0:
                    print("[Bridge] ERROR: ARDY generator unavailable (procedural fallback is disabled).")

            # 3. Pace to ARDY model FPS using wall-clock timing
            now = time.time()
            sleep_duration = next_frame_time - now
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            next_frame_time = max(next_frame_time + frame_interval, time.time())

            if joints is None:
                continue

            # 4. Send frame
            payload = {
                "char_name": current_char_name,
                "frame": frame_num,
                "joints": joints,
                "prompt": current_prompt,
            }
            if rot_list is not None:
                payload["global_rot_mats"] = rot_list

            try:
                packet = json.dumps(payload) + "\n"
                client.sendall(packet.encode("utf-8"))
            except socket.error as e:
                print(f"[Bridge] Send error: {e}")
                running = False
                break

            frame_num += 1

        client.close()
        print("[Bridge Server] Session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender ARDY Real-Time Bridge")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--model", type=str, default="core", help="ARDY model name (e.g. core, soma)")
    parser.add_argument("--quantize-4bit", action="store_true", help="Load text encoder with 4-bit bitsandbytes quantization to save VRAM")
    parser.add_argument("--ardy-dir", type=str, default=None, help="Path to the ARDY root directory")
    args = parser.parse_args()
    run_server(args.port, model_name=args.model, quantize_4bit=args.quantize_4bit)
