import bpy
import os

class CEB_UL_CharacterList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            clean_name = item.name.replace(" ", "_")
            arm_name = item.arm_obj_name if item.arm_obj_name else f"{clean_name}_Armature"
            is_loaded = bpy.data.objects.get(arm_name) is not None

            # Look up parent crowd membership
            parent_crowd = None
            if hasattr(context, "scene") and hasattr(context.scene, "ceb_ardy"):
                for crowd in context.scene.ceb_ardy.crowds:
                    char_names = [n.strip() for n in crowd.character_names.split(",") if n.strip()]
                    if item.name in char_names:
                        parent_crowd = crowd
                        break

            # Requirement 2: If crowd is hidden, disable character selection
            if parent_crowd and parent_crowd.hide_viewport:
                row.enabled = False

            status_icon = 'CHECKMARK' if is_loaded else 'DOT'
            row.label(text="", icon=status_icon)
            row.prop(item, "name", text="", emboss=False)

            # Requirement 1: Show crowd name badge in character selection list
            if parent_crowd:
                crowd_icon = 'HIDE_ON' if parent_crowd.hide_viewport else 'COMMUNITY'
                row.label(text=f"({parent_crowd.name})", icon=crowd_icon)

            row.label(text=item.model.upper())
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)

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

class CEB_UL_CrowdList(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            empty_name = item.empty_object_name if item.empty_object_name else f"ARDY_{item.name}"
            is_loaded = bpy.data.objects.get(empty_name) is not None
            
            status_icon = 'CHECKMARK' if is_loaded else 'DOT'
            row.label(text="", icon=status_icon)
            row.prop(item, "name", text="", emboss=False)

            hide_icon = 'HIDE_ON' if item.hide_viewport else 'HIDE_OFF'
            op = row.operator("ceb.toggle_hide_crowd", text="", icon=hide_icon, emboss=False)
            op.index = index

            row.label(text=f"{item.crowd_count} Chars", icon='COMMUNITY')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.name)

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
        is_ik_active = props.ik_control_active

        # --- ARDY Path Settings ---
        box = layout.box()
        box.label(text="Path Settings", icon='FOLDER_REDIRECT')
        
        package_name = __package__ if __package__ else "CEB_Ardy"
        prefs = context.preferences.addons.get(package_name)
        is_path_set = False

        if prefs:
            box.prop(prefs.preferences, "ardy_path", text="Python Path")
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
        # --- Standalone Viser Web App ---
        box_viser = layout.box()
        box_viser.enabled = is_path_set and (not is_ik_active)
        box_viser.label(text="Viser Web App", icon='URL')
        viser_row = box_viser.row(align=True)
        viser_row.scale_y = 1.1
        viser_row.enabled = is_path_set and (not is_ik_active)
        viser_row.operator("ceb.ardy_run_demo", text="Start Viser Web App", icon='URL')

        # Ensure active character exists
        from .operator import get_active_character, get_active_crowd
        char = get_active_character(context)

        # --- Character Selector ---
        box = layout.box()
        box.enabled = is_path_set
        box.label(text="Character Selection", icon='OUTLINER_OB_ARMATURE')
        
        row = box.row()
        row.enabled = not is_ik_active
        row.template_list(
            "CEB_UL_CharacterList", "",
            props, "characters",
            props, "active_character_index",
            rows=2
        )
        col = row.column(align=True)
        col.enabled = not is_ik_active
        col.operator("ceb.add_character_entry", text="", icon='ADD')
        col.operator("ceb.remove_character_entry", text="", icon='REMOVE')

        if char:
            char_box = box.box()
            char_props_box = char_box.box()
            char_props_box.enabled = not is_ik_active
            char_props_box.prop(char, "name", text="Name")
            char_props_box.prop(char, "model", text="Model")

            # --- IK Control for Character ---
            ik_char_box = char_box.box()
            target_type = getattr(props, "ik_target_type", 'CONSTRAINT')
            if is_ik_active and target_type == 'CHARACTER':
                ik_char_box.alert = True
                ik_char_box.label(text=f"IK Control Active ({props.ik_original_armature_name})", icon='CONSTRAINT_BONE')
                ik_row = ik_char_box.row(align=True)
                ik_row.scale_y = 1.3
                ik_row.operator("ceb.bake_ik_pose", text="Bake Pose to Character", icon='CHECKMARK')
                ik_row.operator("ceb.cancel_ik_control", text="Cancel IK", icon='CANCEL')
            elif not is_ik_active:
                ik_row = ik_char_box.row(align=True)
                ik_row.scale_y = 1.2
                op = ik_row.operator("ceb.start_ik_control", text="IK Control Character", icon='CONSTRAINT_BONE')
                op.target_type = 'CHARACTER'

            # --- Crowd Options (Collapsible) ---
            crowd_main_box = char_box.box()
            crowd_main_box.enabled = not is_ik_active
            
            c_head = crowd_main_box.row(align=True)
            c_icon = 'DISCLOSURE_TRI_DOWN' if props.show_crowd_options else 'DISCLOSURE_TRI_RIGHT'
            c_head.prop(props, "show_crowd_options", text="Crowd Options", icon=c_icon, emboss=False)

            if props.show_crowd_options:
                # --- Crowd Generation (Collapsible) ---
                crowd_gen_box = crowd_main_box.box()
                crowd_gen_box.enabled = not is_ik_active
                cg_head = crowd_gen_box.row(align=True)
                cg_icon = 'DISCLOSURE_TRI_DOWN' if getattr(props, "show_crowd_generation", True) else 'DISCLOSURE_TRI_RIGHT'
                cg_head.prop(props, "show_crowd_generation", text="Crowd Generation", icon=cg_icon, emboss=False)

                if getattr(props, "show_crowd_generation", True):
                    crowd_col = crowd_gen_box.column(align=True)
                    crowd_col.prop(props, "crowd_count", text="Characters Count")
                    
                    layout_row = crowd_col.row(align=True)
                    layout_row.prop(props, "crowd_layout", text="Layout")
                    layout_row.prop(props, "crowd_spacing", text="Spacing (m)")
                    
                    crowd_col.prop(props, "crowd_offset_waypoints", text="Parallel Trajectories (Offset Waypoints)")
                    crowd_col.prop(props, "crowd_reverse_order", text="Simulate From Last Character")
                    
                    avoid_row = crowd_col.row(align=True)
                    avoid_row.prop(props, "crowd_avoid_collisions", text="Avoid Collisions")
                    if props.crowd_avoid_collisions:
                        avoid_row.prop(props, "crowd_avoid_radius", text="Buffer (m)")
                        crowd_col.prop(props, "crowd_avoid_unsimulated", text="Avoid Standing Locations (Unsimulated)")

                    frame_row = crowd_col.row(align=True)
                    frame_row.prop(props, "crowd_start_frame", text="Start Frame")
                    frame_row.prop(props, "crowd_end_frame", text="End Frame")
                    btn_row = crowd_gen_box.row(align=True)
                    btn_row.scale_y = 1.3
                    btn_row.operator("ceb.generate_crowd_animation", text="Generate Crowd Animation", icon='GROUP')

                # --- Crowd Management (Collapsible) ---
                cr_mgmt_box = crowd_main_box.box()
                cr_mgmt_box.enabled = not is_ik_active
                cm_head = cr_mgmt_box.row(align=True)
                cm_icon = 'DISCLOSURE_TRI_DOWN' if getattr(props, "show_crowd_management", True) else 'DISCLOSURE_TRI_RIGHT'
                cm_head.prop(props, "show_crowd_management", text="Crowd Management", icon=cm_icon, emboss=False)

                if getattr(props, "show_crowd_management", True):
                    row = cr_mgmt_box.row()
                    row.enabled = not is_ik_active
                    row.template_list(
                        "CEB_UL_CrowdList", "",
                        props, "crowds",
                        props, "active_crowd_index",
                        rows=2
                    )
                    col = row.column(align=True)
                    col.enabled = not is_ik_active

                    active_crowd = get_active_crowd(context)
                    if active_crowd:
                        c_details = cr_mgmt_box.box()
                        c_details.enabled = not is_ik_active
                        c_details.prop(active_crowd, "name", text="Name")
                        if active_crowd.empty_object_name:
                            c_details.label(text=f"Parent Empty: {active_crowd.empty_object_name}", icon='EMPTY_DATA')

                        # c_details.prop(active_crowd, "clear_settings_before_generate", text="Clear Settings Before Generate")

                        c_btns = c_details.row(align=True)
                        c_btns.scale_y = 1.2
                        c_btns.operator("ceb.select_crowd", text="Select", icon='RESTRICT_SELECT_OFF')
                        # c_btns.operator("ceb.regenerate_crowd", text="Regenerate", icon='FILE_REFRESH')
                        c_btns.operator("ceb.clear_crowd_settings", text="Clear Settings", icon='TRASH')
                        c_btns.operator("ceb.delete_crowd", text="Delete", icon='TRASH')

        else:
            char_box = box.box()
            char_box.enabled = not is_ik_active
            char_box.label(text="Click '+' to add a character", icon='INFO')

        # --- Real-Time Control ---
        box = layout.box()
        box.enabled = is_path_set and (not is_ik_active)
        box.label(text="Real-Time Control", icon='ORIENTATION_PARENT')
        
        is_connected = (props.realtime_status == "Connected")
        status_icon = 'CHECKMARK' if is_connected else 'ERROR'
        
        row = box.row()
        row.label(text=f"Status: {props.realtime_status}", icon=status_icon)
        row.prop(props, "realtime_port", text="Port")
        
        box.prop(props, "quantize_4bit", text="4-bit Quantization (VRAM Save)")
        box.prop(props, "save_as_nla", text="Save as NLA Track")
        
        mute_row = box.row()
        mute_row.enabled = props.save_as_nla
        mute_row.prop(props, "mute_previous_nla_layers", text="Mute Previous NLA Layers")
        
        row = box.row(align=True)
        row.operator("ceb.clean_animation", text="Clean Animation", icon='TRASH')
        
        # Bridge Server Controls
        row = box.row(align=True)
        row.scale_y = 1.2
        row.enabled = is_path_set and (not is_ik_active)
        row.operator("ceb.ardy_start_bridge", text="Start Bridge", icon='PLAY')

        # Connect / Disconnect Stream Controls (Fixed position, depress highlight when connected)
        stream_text = "Disconnect Stream" if is_connected else "Connect Stream"
        stream_icon = 'CANCEL' if is_connected else 'LINK_BLEND'
            
        stream_row = box.row(align=True)
        stream_row.scale_y = 1.5
        stream_row.enabled = is_path_set and (not is_ik_active)
        stream_row.operator("ceb.ardy_realtime_stream", text=stream_text, icon=stream_icon, depress=is_connected)
        
        # Show Live Prompt ONLY if stream is connected and prompt schedule list is empty
        if is_connected and char and len(char.prompt_schedule) == 0:
            box.prop(char, "realtime_prompt", text=f"Live Prompt ({char.name})")

        # --- Prompt Schedule ---
        if char:
            box = layout.box()
            box.enabled = is_path_set
            header = box.row(align=True)
            header.enabled = not is_ik_active
            header.label(text=f"Prompt Schedule ({char.name})", icon='TIME')
            header.prop(props, "overlay_view_mode", text="")
            header.prop(props, "show_prompt_overlay", text="", icon='OVERLAY')
            
            row = box.row()
            row.enabled = not is_ik_active
            row.template_list(
                "CEB_UL_PromptList", "",
                char, "prompt_schedule",
                char, "prompt_schedule_index",
                rows=3
            )
            
            col = row.column(align=True)
            col.enabled = not is_ik_active
            col.operator("ceb.add_prompt_item", text="", icon='ADD')
            col.operator("ceb.add_waypoint", text="", icon='ORIENTATION_LOCAL')
            col.operator("ceb.add_pose_constraint", text="", icon='ARMATURE_DATA')
            col.operator("ceb.capture_pose_constraint", text="", icon='POSE_HLT')
            col.operator("ceb.remove_prompt_item", text="", icon='REMOVE')
            col.operator("ceb.clear_prompt_items", text="", icon='TRASH')
            col.separator()
            col.operator("ceb.sort_prompt_items", text="", icon='SORTTIME')
            col.separator()
            col.operator("ceb.move_prompt_item", text="", icon='TRIA_UP').direction = 'UP'
            col.operator("ceb.move_prompt_item", text="", icon='TRIA_DOWN').direction = 'DOWN'

            # Selected Prompt / Waypoint / Pose Details Box
            if 0 <= char.prompt_schedule_index < len(char.prompt_schedule):
                selected_item = char.prompt_schedule[char.prompt_schedule_index]
                wp_box = box.box()
                if selected_item.has_waypoint:
                    wp_box.enabled = not is_ik_active
                    wp_row = wp_box.row(align=True)
                    wp_row.prop(selected_item, "has_waypoint", text="3D Waypoint Target")
                    wp_row.operator("ceb.remove_waypoint", text="", icon='X')
                    loc_row = wp_box.row(align=True)
                    loc_row.prop(selected_item, "waypoint_co", text="Location")
                    if selected_item.waypoint_object_name:
                        wp_box.label(text=f"Linked Empty: {selected_item.waypoint_object_name}", icon='EMPTY_DATA')
                elif getattr(selected_item, "has_pose_constraint", False):
                    pc_row = wp_box.row(align=True)
                    pc_row.enabled = not is_ik_active
                    pc_row.prop(selected_item, "has_pose_constraint", text="Pose Constraint Target")
                    pc_row.operator("ceb.capture_pose_constraint", text="Re-capture Pose", icon='POSE_HLT')
                    pc_row.operator("ceb.remove_pose_constraint", text="", icon='X')
                    if selected_item.pose_armature_name:
                        wp_box.label(text=f"Target Armature: {selected_item.pose_armature_name}", icon='POSE_HLT')
                        
                        ik_box = wp_box.box()
                        if is_ik_active and target_type == 'CONSTRAINT':
                            ik_box.alert = True
                            ik_box.label(text=f"IK Control Active ({props.ik_original_armature_name})", icon='CONSTRAINT_BONE')
                            ik_row = ik_box.row(align=True)
                            ik_row.scale_y = 1.3
                            ik_row.operator("ceb.bake_ik_pose", text="Bake Pose to Constraint", icon='CHECKMARK')
                            ik_row.operator("ceb.cancel_ik_control", text="Cancel IK", icon='CANCEL')
                        elif not is_ik_active:
                            ik_row = ik_box.row(align=True)
                            ik_row.scale_y = 1.2
                            op = ik_row.operator("ceb.start_ik_control", text="IK Control", icon='CONSTRAINT_BONE')
                            op.target_type = 'CONSTRAINT'


classes = (
    CEB_UL_CharacterList,
    CEB_UL_PromptList,
    CEB_UL_CrowdList,
    CEB_PT_ArdyPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
