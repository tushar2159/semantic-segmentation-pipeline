import torch
from segmentation_pipeline.model import TinyUNet
from segmentation_pipeline.metrics import mean_iou
def test_model_contract():
    y=TinyUNet(num_classes=4)(torch.randn(2,3,64,64))
    assert y.shape==(2,4,64,64)
def test_perfect_iou():
    target=torch.tensor([[[0,1],[1,0]]])
    logits=torch.nn.functional.one_hot(target,2).permute(0,3,1,2).float()*10
    assert mean_iou(logits,target,2)==1.0
