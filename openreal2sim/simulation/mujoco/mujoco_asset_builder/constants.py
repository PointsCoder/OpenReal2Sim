# XML formatting
XML_INDENTATION = "  "

# MJCF class names
CLASS_VISUAL = "visual"
CLASS_COLLISION = "collision"
CLASS_OBJ_VISUAL = "obj_visual"
CLASS_OBJ_COLLISION = "obj_collision"

# Geometry groups
GROUP_VISUAL = "2"
GROUP_COLLISION = "3"

# Collision contact parameters for asset builder
COLLISION_MARGIN = "0.006"
COLLISION_SOLREF = "0.0005 1"
COLLISION_SOLIMP = "0.998 0.9999 0.0005"

# Object collision parameters (runtime fusion)
OBJ_COLLISION_MARGIN = "0.004"
OBJ_COLLISION_SOLREF = "0.004 1"
OBJ_COLLISION_SOLIMP = "0.95 0.99 0.001"

# Joint parameters
JOINT_TYPE_FREE = "free"
FREEJOINT_DAMPING = "0.1"
OBJ_FREEJOINT_DAMPING = "0.5"

# Geometry types
GEOM_TYPE_MESH = "mesh"
CONTYPE_NONE = "0"
CONAFFINITY_NONE = "0"

# Simulation parameters
DEFAULT_TIMESTEP = "0.001"
DEFAULT_MEMORY = "64M"

# Material parameters
MATERIAL_SPECULAR = "0.4"
MATERIAL_SHININESS = "0.001"

# Default physical parameters
DEFAULT_Z_OFFSET = 0.005
DEFAULT_INERTIA_SCALE = 0.002

# File naming conventions
SUFFIX_OPTIMIZED = "_optimized"
SUFFIX_COLLISION = "_collision_"
SUFFIX_METADATA = "_metadata.json"
MATERIAL_PREFIX = "material_0"

# File extensions
EXT_GLB = ".glb"
EXT_GLTF = ".gltf"
EXT_OBJ = ".obj"
EXT_MTL = ".mtl"
EXT_XML = ".xml"
EXT_JSON = ".json"
EXT_PNG = ".png"
EXT_JPG = ".jpg"
EXT_JPEG = ".jpeg"

# Asset type detection
ASSET_TYPE_BACKGROUND = "background"
ASSET_TYPE_OBJECT = "object"

# Standard file names
FILENAME_SCENE = "scene.json"
FILENAME_TRAJECTORY = "trajectory.json"
FILENAME_PANDA_CONFIG = "franka_panda_config.json"
FILENAME_OBJECT_MASSES = "object_masses.json"
FILENAME_SCENE_XML = "scene.xml"
FILENAME_PANDA_XML = "panda.xml"

# Standard directory names
DIR_MJCF_ASSETS = "mjcf_assets"
DIR_ROBOTS = "assets/robots/franka_emika_panda"
DIR_CONFIG = "config"

# Mesh compiler settings
MESHDIR_VFS = "."
TEXTUREDIR_VFS = "."

# MTL file processing
MTL_COMMENT_CHAR = "#"
MTL_FIELDS = (
    "Ka",
    "Kd",
    "Ks",
    "d",
    "Tr",
    "Ns",
    "map_Kd",
)
