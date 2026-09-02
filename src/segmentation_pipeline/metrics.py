def mean_iou(logits, target, num_classes):
    pred = logits.argmax(1)
    vals = []
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        if union:
            vals.append((p & t).sum().item() / union)
    return float(sum(vals) / max(len(vals), 1))
