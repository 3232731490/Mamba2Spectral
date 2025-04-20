from .yolov8CSPDarknetBackbone import build_yoloV8CSPDarknetStage
from .sparXMamba import build_sparXMambaStage

all_backbones = {
    "yoloV8CSPDarknet": build_yoloV8CSPDarknetStage,
    'sparXMamba': build_sparXMambaStage,
}