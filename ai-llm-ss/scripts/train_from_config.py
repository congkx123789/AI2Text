import json, subprocess, sys, os

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/train_merged.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    def add_flag(args, name, value):
        if isinstance(value, bool):
            if value:
                args.append(f"--{name}")
        else:
            args.extend([f"--{name}", str(value)])

    args = []
    for k, v in cfg.items():
        # keep ordering roughly stable for readability
        if k in ("manifest","audio_root","timestamps","vocab"):
            add_flag(args, k, v)
    for k in ("trim_segments","device","amp","batch_size","gradient_accumulation_steps","max_grad_norm","empty_cache","num_workers","epochs","log_interval","lr","checkpoint_dir","resume","out"):
        if k in cfg:
            add_flag(args, k, cfg[k])

    cmd = [sys.executable, "-m", "src.asr.train_ctc"] + args
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    raise SystemExit(subprocess.call(cmd, env=env))

if __name__ == "__main__":
    main()

