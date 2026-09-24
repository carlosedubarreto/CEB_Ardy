# CEB Ardy - Real-Time AI Motion Generation for Blender

[![Blender Version](https://img.shields.io/badge/Blender-4.4%20%7C%204.5%2B-orange?logo=blender&logoColor=white)](https://www.blender.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Powered%20By-NVIDIA%20ARDY-76B900?logo=nvidia&logoColor=white)](https://github.com/)
[![Version](https://img.shields.io/badge/Version-1.1.0-brightgreen)]()
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows)]()

**CEB Ardy** is a Blender add-on developed by **Carlos Barreto** that integrates NVIDIA's **ARDY** (Autoregressive Diffusion for real-time human motion generation) framework directly into the Blender 3D Viewport.

Generate, steer, condition, and layer realistic human character animations in real time using text prompts, 3D spatial waypoints, custom pose constraints, and procedural crowd simulation tools.

---

## 🌟 Key Features

### ⚡ Real-Time Motion Generation
- **Bi-directional Bridge**: Live TCP streaming bridge connects Blender directly to the ARDY generative model.
- **Interactive Text Prompting**: Drive animations live or execute sequenced prompt schedules (e.g. *walk*, *crouch*, *wave*, *run*, *jump*).
- **3D Viewport Prompt Overlay**: Real-time HUD displaying active prompts and timing directly inside the 3D Viewport.
- **4-Bit Quantization**: Optional 4-bit quantization mode to drastically reduce VRAM consumption while preserving motion fidelity.

### 👥 Procedural Crowd Simulation
- **Multi-Character Management**: Create, configure, and animate multiple characters simultaneously in the same scene.
- **Spatial Layout Patterns**: Generate crowds arranged in **Grid**, **Line**, **Circle**, or **Random Scatter** formations with customizable spacing.
- **Dynamic Collision Avoidance**: Automatically calculates detour waypoints so moving agents steer clear of previously simulated characters and standing agents.
- **Parallel Trajectories**: Automatically offsets waypoints relative to each character's origin to maintain group formations without lane crossover.
- **Crowd Hierarchy**: Organizes crowd characters under dedicated parent empties with one-click viewport visibility toggles, batch selection, and management.

### 🎯 Spatial Waypoints & Steering
- **3D Waypoint Empties**: Place interactive empties anywhere in 3D space and bind them to specific timeline frames.
- **Dynamic Trajectory Following**: Characters naturally steer, turn, and navigate toward 3D waypoints according to active text commands.

### 🦴 Pose Constraints & Interactive IK Rig
- **Keyframe Pose Conditioning**: Enforce specific character poses at precise frames (e.g., stopping at an exact stance or touching an object).
- **Built-in IK Control Rig**: Temporarily spawn an IK control rig (`Ardy_Core_Rig` / `Ardy_Core_Rig_Character`) to pose characters or constraints with intuitive Blender IK handles.
- **One-Click Pose Baking**: Seamlessly bake custom IK poses back into the ARDY pose constraint or character skeleton.

### 🎬 Production & Retargeting Pipeline
- **Auto NLA Layering**: Automatically records each generation run into organized Non-Linear Animation (NLA) strips.
- **Actions Manager**: View, assign, unassign, and audit all generated motion actions per character.
- **Retargeting Friendly**: Designed for smooth retargeting to standard pipelines such as Auto-Rig Pro (ARP) or game engine skeletons (Unreal / Unity).
- **Viser Web App Launcher**: Launch the standalone ARDY Viser Web App directly from the Blender N-panel.

---

## 📋 Requirements

- **Blender**: 4.4.0 or newer (tested with Blender 4.4 and 4.5)
- **OS**: Windows 10 / 11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA support (RTX series recommended)
- **ARDY Runtime**: A portable Python environment (Python 3.11) or virtual environment containing:
  - PyTorch (CUDA-enabled)
  - ARDY repository & model checkpoints
  - Required dependencies (`numpy`, `torch`, `einops`, `viser`, etc.)

---

## 🚀 Installation & Setup

### 1. Install the Addon in Blender
1. Download or package the `CEB_Ardy` folder as a `.zip` archive (or place `CEB_Ardy` directly inside your Blender addons folder: `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\CEB_Ardy`).
2. In Blender, navigate to **Edit > Preferences > Add-ons**.
3. Click **Install...**, select the `.zip` or enable **Animation: CEB Ardy**.

### 2. Configure Addon Preferences
1. In **Preferences > Add-ons > CEB Ardy**, set the **Portable Python Folder**:
   - Point to the folder containing `python.exe` (e.g., `.../Portable_Ardy/python-3.11.9-embed-amd64`) **or** the parent folder containing both your Python directory and the `ardy` folder.
2. Ensure the status indicator confirms:
   ```
   ✔ Python and ARDY detected successfully!
   ```

### 3. Start the Server
1. Open the **3D Viewport** sidebar (**N** key) and select the **CEB** tab.
2. In the **Real-Time Control** section, check the server port (default: `9999`).
3. Click **Start Server**. The status will display:
   ```
   Server: Running
   ```

---

## 📖 Quick Start Workflow

### A. Single Character Live Stream
1. Click **`+`** under **Character Selection** to add a character (`Char_1`). The character armature and mesh load automatically.
2. In **Real-Time Control**, toggle **4-bit Quantization** if you wish to conserve VRAM.
3. Click **Connect Stream**. Blender connects to the ARDY bridge server.
4. Type prompts into the **Live Prompt** box (e.g., `walk forward`, `jog`, `crouch walk`, `dance`) and watch the character respond in real time.
5. Click **Disconnect Stream** to stop. The animation is automatically saved into Blender's NLA editor.

### B. Prompt Schedule & Spatial Waypoints
1. Select your character in the **Character Selection** list.
2. In the **Prompt Schedule** sub-panel:
   - Click **`+`** to add a prompt item, set its **Frame** (e.g., frame 1: `walk forward`).
   - Click the **Waypoint** icon to add a 3D target at frame 60. An Empty will appear in the viewport.
   - Move the waypoint Empty to where you want the character to walk.
   - Add another prompt at frame 100 (e.g., `turn right and run`).
3. Connect the stream or run simulation to watch the character follow the timed prompts and steer toward the waypoints.

### C. IK Control & Custom Pose Constraints
1. In the Prompt Schedule, click **Add Pose Constraint** (Armature icon) at a target frame.
2. Click **IK Control**. The interactive IK control rig loads automatically, hiding the base mesh to let you pose comfortably.
3. Move the IK handles (hands, feet, spine, root) to formulate the target pose.
4. Click **Bake Pose to Constraint** (or **Bake Pose to Character**).
5. ARDY will naturally blend and pass through this exact pose at the scheduled frame!

### D. Crowd Generation
1. In **Character Selection > Crowd Options**, expand **Crowd Generation**:
   - **Characters Count**: Set how many characters to spawn (e.g., `10`).
   - **Layout**: Choose `GRID`, `LINE`, `CIRCLE`, or `RANDOM`.
   - **Spacing (m)**: Distance between characters (e.g., `2.5m`).
   - **Avoid Collisions**: Enable to automatically calculate steering detours preventing overlapping agent paths.
   - **Parallel Trajectories**: Enable so characters follow synchronized paths offset by their starting positions.
2. Click **Generate Crowd Animation**.
3. All agents are generated, organized into a dedicated Crowd group under a parent Empty, and simulated sequentially.
4. Use **Crowd Management** to hide/unhide crowds or clean up scene data in one click.

---

## 🗂️ Panel Guide (N-Panel > CEB)

| Section | Description |
| :--- | :--- |
| **Path Settings** | Configure and verify the ARDY environment and Python runtime. |
| **Viser Web App** | Quick launcher for the official ARDY browser-based GUI. |
| **Character Selection** | Add, remove, and select active characters; access IK control and crowd options. |
| **Crowd Options** | Procedural multi-character spawning, layout patterns, collision avoidance, and crowd management. |
| **Available Actions** | Library of all generated motion actions. Quick assign/unassign to the active armature for comparison and retargeting. |
| **Real-Time Control** | Start/stop bridge server, port settings, 4-bit quantization, connect/disconnect live stream, and clean animation data. |
| **Prompt Schedule** | Frame-by-frame text prompt scheduling, 3D waypoint creation, pose constraint capture, and viewport overlay toggles. |

---

## 🛠️ Retargeting & Production Tips

- **Auto-Rig Pro (ARP)**: Use the included remap preset (`remap_preset_ARDY_to_ARP.bmap`) or standard Quick Rig mapping to retarget generated ARDY motions onto custom characters and game-ready rigs.
- **NLA Strips**: Each generation pass automatically creates an NLA strip. You can mute previous layers or blend multiple generated motions across the timeline.
- **Performance**: When generating large crowds (20+ characters), keep **4-bit Quantization** enabled and close unnecessary background applications to maximize GPU throughput.

---

## 📁 Repository Structure

```
CEB_Ardy/
├── __init__.py                     # Addon initialization, metadata, and preferences
├── panel.py                        # 3D Viewport N-Panel UI and UIList definitions
├── operator.py                     # Core operators, socket bridge client, IK solvers, crowd generation
├── blender_bridge.py               # Standalone Python bridge server interfacing directly with ARDY
├── Ardy_Core_Rig.blend             # Base IK control rig for pose constraints
└── Ardy_Core_Rig_Character.blend   # Full IK character rig for interactive character posing
```

---

## 👤 Author & Acknowledgments

- **Author**: [Carlos Barreto](https://github.com/)
- **Technology**: Built upon the [NVIDIA ARDY](https://github.com/) Real-Time Motion Generation Framework.

---

## 📄 License

This project is distributed under the terms of the project repository license. Please refer to NVIDIA's ARDY repository license for model weights, code, and underlying generative framework terms of use.
