"""Count trainable parameters without loading pretrained weights."""
import argparse
import network.modeling as modeling

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=modeling.MODEL_NAMES, default='m2f_deeplabv3plus_xception16')
    parser.add_argument('--output_stride', type=int, choices=(8, 16), default=16)
    opts = parser.parse_args()
    model = getattr(modeling, opts.model)(num_classes=2, output_stride=opts.output_stride, pretrained_backbone=False)
    print(f'{opts.model}: {sum(p.numel() for p in model.parameters()):,} parameters')
