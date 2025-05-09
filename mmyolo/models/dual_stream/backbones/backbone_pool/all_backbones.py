from .yolov8CSPDarknetBackbone import build_yoloV8CSPDarknetStage
from .sparXMamba import build_sparXMambaStage
from .yolov11Backbone import build_yoloV11Stage
from .mobileMamba import build_mobile_mambe_stage
# from .mamba_vision import build_mambavision_stage

all_backbones = {
    "yoloV8CSPDarknet": build_yoloV8CSPDarknetStage,
    'sparXMamba': build_sparXMambaStage,
    'yolov11Backbone': build_yoloV11Stage,
    'mobileMamba': build_mobile_mambe_stage,
    # 'mamba_vision': build_mambavision_stage,
}