bl_info = {
    "name": "CEB Ardy",
    "author": "Carlos Barreto",
    "version": (1, 1, 0),
    "blender": (4, 4, 0),
    "location": "3D Viewport > Sidebar (N-panel) > CEB",
    "description": "Integrate NVIDIA ARDY real-time motion generation framework into Blender",
    "category": "Animation",
}

### OK - 4 bit quantization
### OK - List of prompts
###        OK - Esta demorando muito para poder agir, tentar ajustar para ser mais rapido
### OK - clear animation (blender e blender bridge)
### OK - add waypoints
### OK - add constraints
### OK - add option to get the current pose and add as constraint
### OK - multiple characters
### OK - autosave animation as NLA strip at each run
### OK - portable python enable quantization webapp

### WIP 4.1
### OK - retirar o empty quando carregar o personagem
### OK - ajustar para a simulatcao funcionar mesmo se o character nao estiver no centro do 3dview

### wip5
### OK - update para blender 5.2 
### OK - Temp Ik to control the pose easily
### OK - consertar ik control para o character quando em esta em posição diferente da padrao, emuma curva, por exemplo

### wip6 i
### - keyboard based velocity
### - Kinematic constraints 

### - retarget from sam 3d  body result (nao para agora, tive muitos problemas)
### - list of nla strip recorded (name is the prompts) (TALVEZ) , pois ja tenho essa função no SAM

#---------------------------------
## V1.01
## OK - adicionar crowd generation, colocar parametros para adicionar N quantidade de personagens e gerar uma animação para cada um deles, usando os mesmos prompts

## V1.01 Wip2
## OK - Adicionar opção para que personagens nao atravessem outros.
## - Criar list com os crowds
## - Opção para regerar, recriando reganrando as opções ativas par ao crowd ou ignorando a geração dos avoid
## - Gerenciamento dos crows, e opçao de esconder
## - 



import bpy
from bpy.props import StringProperty

# Support reloading submodules
if "bpy" in locals():
    import importlib
    if "panel" in locals():
        importlib.reload(panel)
    if "operator" in locals():
        importlib.reload(operator)

from . import panel
from . import operator

class CEB_Ardy_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__ if __package__ else "CEB_Ardy"

    ardy_path: StringProperty(
        name="Portable Python Folder",
        description="Select the portable Python folder (the 'ardy' folder should be a sibling)",
        subtype='DIR_PATH',
        default=""
    )

    def draw(self, context):
        layout = self.layout
        column = layout.column(align=True)
        column.prop(self, "ardy_path")
        
        # Add some quick help/status in preferences
        if not self.ardy_path:
            column.label(text="Please select the portable Python folder.", icon='ERROR')
        else:
            from .operator import get_ardy_paths
            paths, err = get_ardy_paths(context)
            if err:
                column.label(text=err, icon='WARNING')
            else:
                column.label(text="Python and ARDY detected successfully!", icon='CHECKMARK')

classes = (
    CEB_Ardy_Preferences,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    operator.register()
    panel.register()

def unregister():
    panel.unregister()
    operator.unregister()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
