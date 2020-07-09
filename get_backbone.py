from detectron2.modeling.backbone import Backbone
from detectron2.layers import FrozenBatchNorm2d, ShapeSpec, get_norm
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

! git clone https://github.com/youngwanLEE/vovnet-detectron2


@BACKBONE_REGISTRY.register()
def build_vovnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_vovnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

from detectron2.config import CfgNode as CN

def add_vovnet_config(cfg):
    """
    Add config for VoVNet.
    """
    _C = cfg

    _C.MODEL.VOVNET = CN()

    _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VOVNET.NORM = "FrozenBN"

    _C.MODEL.VOVNET.OUT_CHANNELS = 256

    _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256


cfg = get_cfg()
add_vovnet_config(cfg)
cfg.merge_from_file('vovnet-detectron2/configs/mask_rcnn_V_57_FPN_3x.yaml')[]