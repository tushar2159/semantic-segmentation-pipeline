import argparse
import json
import torch
from .evaluate import evaluate

def main():
    p=argparse.ArgumentParser(prog="segmentation-pipeline")
    p.add_argument("command",choices=["prepare","train","evaluate","predict","self-check"])
    p.add_argument("--config",default="config/default.yaml")
    args=p.parse_args()
    if args.command=="prepare":
        print("Synthetic/public-safe demo data requires no download.")
    elif args.command=="train":
        from .train import train
        print(train(args.config))
    elif args.command=="evaluate":
        print(json.dumps(evaluate(args.config),indent=2))
    elif args.command=="predict":
        print("Use the package predict_tensor/build_demo_router API for tensor-level inference.")
    elif args.command=="self-check":
        result=evaluate(args.config)
        if not result: raise SystemExit("self-check failed")
        print(json.dumps({"status":"ok","evaluation":result},indent=2))

if __name__=="__main__":
    main()
