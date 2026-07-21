import bpy
import os

class CEB_UL_PromptList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="", icon='CHECKBOX_HLT' if item.enabled else 'CHECKBOX_DEHLT', emboss=False)
            row.prop(item, "start_frame", text="Frame", emboss=True)
            if getattr(item, "has_waypoint", False):
                row.label(text="Waypoint", icon='ORIENTATION_LOCAL')
                op = row.operator("ceb.select_prompt_item", text="", icon='RESTRICT_SELECT_OFF', emboss=False)
                op.index = index
            elif getattr(item, "has_pose_constraint", False):
                row.label(text="Pose Constraint", icon='ARMATURE_DATA')
                op = row.operator("ceb.select_prompt_item", text="", icon='RESTRICT_SELECT_OFF', emboss=False)
                op.index = index
            else:
                row.prop(item, "prompt", text="", emboss=True)
                op = row.operator("ceb.select_prompt_item", text="", icon='RESTRICT_SELECT_OFF', emboss=False)
                op.index = index
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=str(item.start_frame))

class CEB_PT_ArdyPanel(bpy.types.Panel):
    bl_label = "CEB ARDY"
    bl_idname = "CEB_PT_ardypanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Safety check: ensure properties are initialized
        if not hasattr(scene, "ceb_ardy"):
            layout.label(text="Addon registration error: scene properties missing.", icon='ERROR')
            return
            
        props = scene.ceb_ardy

        # --- ARDY Path Settings ---
        box = layout.box()
        box.label(text="Path Settings", icon='FOLDER_REDIRECT')
        
        # Access the addon preferences
        package_name = __package__ if __package__ else "CEB_Ardy"
        prefs = context.preferences.addons.get(package_name)
        is_path_set = False

        if prefs:
            box.prop(prefs.preferences, "ardy_path", text="Python Path")
            
            # Check validation status in UI
            if not prefs.preferences.ardy_path:
                row = box.row()
                row.label(text="Path not configured!", icon='ERROR')
            else:
                from .operator import get_ardy_paths
                paths, err = get_ardy_paths(context)
                if err:
                    row = box.row()
                    row.label(text=err, icon='WARNING')
                else:
                    is_path_set = True
                    row = box.row()
                    row.label(text="Python & ARDY detected", icon='CHECKMARK')
        else:
            box.label(text="Addon preferences not registered.", icon='ERROR')

        # --- Real-Time Control ---
        box = layout.box()
        box.label(text="Real-Time Control", icon='ORIENTATION_PARENT')
        
        status_icon = 'ERROR' if props.realtime_status == "Disconnected" else 'CHECKMARK'
        row = box.row()
        row.label(text=f"Status: {props.realtime_status}", icon=status_icon)
        row.prop(props, "realtime_port", text="Port")
        
        box.prop(props, "model", text="Model")
        box.prop(props, "quantize_4bit", text="4-bit Quantization (VRAM Save)")
        
        row = box.row(align=True)
        row.prop(props, "realtime_recording", text="Record Animation")
        row.operator("ceb.clean_animation", text="Clean Animation", icon='TRASH')
        
        row = box.row(align=True)
        row.scale_y = 1.1
        row.enabled = is_path_set or (props.realtime_status == "Connected")
        row.operator("ceb.ardy_start_bridge", text="Start Bridge", icon='PLAY')
        
        stream_text = "Disconnect Stream" if props.realtime_status == "Connected" else "Connect Stream"
        stream_icon = 'CANCEL' if props.realtime_status == "Connected" else 'LINK_BLEND'
        row.operator("ceb.ardy_realtime_stream", text=stream_text, icon=stream_icon)
        
        if props.realtime_status == "Connected":
            box.prop(props, "realtime_prompt")

        # --- Prompt Schedule ---
        box = layout.box()
        header = box.row(align=True)
        header.label(text="Prompt Schedule", icon='TIME')
        header.prop(props, "show_prompt_overlay", text="", icon='OVERLAY')
        
        row = box.row()
        row.template_list(
            "CEB_UL_PromptList", "",
            props, "prompt_schedule",
            props, "prompt_schedule_index",
            rows=3
        )
        
        col = row.column(align=True)
        col.operator("ceb.add_prompt_item", text="", icon='ADD')
        col.operator("ceb.add_waypoint", text="", icon='ORIENTATION_LOCAL')
        col.operator("ceb.add_pose_constraint", text="", icon='ARMATURE_DATA')
        col.operator("ceb.remove_prompt_item", text="", icon='REMOVE')
        col.operator("ceb.clear_prompt_items", text="", icon='TRASH')
        col.separator()
        col.operator("ceb.sort_prompt_items", text="", icon='SORTTIME')
        col.separator()
        col.operator("ceb.move_prompt_item", text="", icon='TRIA_UP').direction = 'UP'
        col.operator("ceb.move_prompt_item", text="", icon='TRIA_DOWN').direction = 'DOWN'

        # Selected Prompt / Waypoint / Pose Details Box
        if 0 <= props.prompt_schedule_index < len(props.prompt_schedule):
            selected_item = props.prompt_schedule[props.prompt_schedule_index]
            wp_box = box.box()
            if selected_item.has_waypoint:
                wp_row = wp_box.row(align=True)
                wp_row.prop(selected_item, "has_waypoint", text="3D Waypoint Target")
                wp_row.operator("ceb.remove_waypoint", text="", icon='X')
                loc_row = wp_box.row(align=True)
                loc_row.prop(selected_item, "waypoint_co", text="Location")
                if selected_item.waypoint_object_name:
                    wp_box.label(text=f"Linked Empty: {selected_item.waypoint_object_name}", icon='EMPTY_DATA')
            elif getattr(selected_item, "has_pose_constraint", False):
                pc_row = wp_box.row(align=True)
                pc_row.prop(selected_item, "has_pose_constraint", text="Pose Constraint Target")
                pc_row.operator("ceb.remove_pose_constraint", text="", icon='X')
                if selected_item.pose_armature_name:
                    wp_box.label(text=f"Target Armature: {selected_item.pose_armature_name}", icon='POSE_HLT')

classes = (
    CEB_UL_PromptList,
    CEB_PT_ArdyPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
