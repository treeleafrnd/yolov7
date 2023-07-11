def postprocess(fpn_heads_outputs,conf_thres,max_detections,multiple_labels_per_box: bool = True)

    preds = derive_preds(fpn_heads_outputs)
    formatted_preds = format_preds(
        preds, conf_thres, max_detections, multiple_labels_per_box
    )
    return formatted_preds

def _derive_preds( fpn_heads_outputs):
    all_preds = []
    for layer_idx, fpn_head_outputs in enumerate(
            fpn_heads_outputs[: detection_head.num_layers]
    ):
        batch_size, _, num_rows, num_cols, *_ = fpn_head_outputs.shape
        grid = self._make_grid(num_rows, num_cols).to(fpn_head_outputs.device)
        fpn_head_preds = transform_model_outputs_into_predictions(fpn_head_outputs)
        fpn_head_preds[
            ..., [PredIdx.CY, PredIdx.CX]
        ] += grid  # Grid corrections -> Grid coordinates
        fpn_head_preds[
            ..., [PredIdx.CX, PredIdx.CY]
        ] *= self.detection_head.strides[
            layer_idx
        ]  # -> Image coordinates
        # TODO: Probably can do it in a more standardized way
        fpn_head_preds[
            ..., [PredIdx.W, PredIdx.H]
        ] *= self.detection_head.anchor_grid[
            layer_idx
        ]  # Anchor box corrections -> Image coordinates
        fpn_head_preds[..., PredIdx.OBJ:].sigmoid_()
        all_preds.append(
            fpn_head_preds.view(batch_size, -1, self.detection_head.num_outputs)
        )
    return torch.cat(all_preds, 1)
