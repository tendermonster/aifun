# pytorch_total_params = sum(p.numel() for p in model.parameters())
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_num_parameters(model, include_grad=False) -> int:
    if include_grad:
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params
