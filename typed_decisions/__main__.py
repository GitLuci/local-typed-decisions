import argparse
import json
import sys
from pathlib import Path

from .runtime import LocalDecisionModel

serve_command = sys.argv[1:2] == ["serve"]
argv = ["--serve", *sys.argv[2:]] if serve_command else sys.argv[1:]
parser = argparse.ArgumentParser(description="Local, offline typed decisions; serve --config FILE starts the GGUF API")
parser.add_argument("request", nargs="?", type=Path)
parser.add_argument("--serve", action="store_true")
parser.add_argument("--port", type=int, default=8788)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--backend", choices=["laya", "qwen", "llama-server"], default="llama-server" if serve_command else "laya")
parser.add_argument("--mode", choices=["ultra", "fast", "medium", "slow", "routed"], help="Override the llama-server config mode")
parser.add_argument("--config", type=Path, help="Local llama-server JSON: ultra|fast|medium|slow|routed")
parser.add_argument("--qwen-lock", type=Path, help="Pinned model lock; e.g. qwen-1.7b.lock.json")
parser.add_argument("--readout", choices=["letter", "json"], default="letter")
parser.add_argument("--choice-order", choices=["input", "canonical"], default="input")
parser.add_argument("--consistency-rounds", type=int, default=0, help="Optional Choice permutations in one forward (CLI only)")
args = parser.parse_args(argv)
if serve_command and (args.backend != "llama-server" or not args.config or args.request):
    parser.error("serve requires --config and the llama-server backend, without a request file")
if not 1 <= args.port <= 65535:
    parser.error("--port must be in [1, 65535]")
llama_options = None
if args.config:
    if args.backend != "llama-server":
        parser.error("--config requires --backend llama-server")
    from .llama_server import load_config
    llama_options = load_config(args.config)
if args.mode:
    if args.backend != "llama-server" or llama_options is None:
        parser.error("--mode requires --backend llama-server and --config")
    llama_options["mode"] = args.mode
    if args.mode != "routed":
        llama_options.pop("routes", None)
        llama_options.pop("default_domain", None)
if args.consistency_rounds and (args.backend != "qwen" or args.serve):
    parser.error("--consistency-rounds requires a Qwen CLI request")
if args.serve:
    import uvicorn
    from .server import create_app
    uvicorn.run(create_app(threads=args.threads, backend=args.backend,
                          qwen_options={"lock_file":args.qwen_lock,"readout":args.readout,"choice_order":args.choice_order},
                          llama_options=llama_options),
                host="127.0.0.1", port=args.port)
elif args.request:
    payload = json.loads(args.request.read_text(encoding="utf-8-sig"))
    if args.backend == "llama-server":
        from .llama_server import create_backend
        model = create_backend(**(llama_options or {}))
    elif args.backend == "qwen":
        from .qwen import QwenDecisionModel
        model = QwenDecisionModel(threads=args.threads, lock_file=args.qwen_lock,
                                  readout=args.readout,choice_order=args.choice_order)
    else:
        model = LocalDecisionModel(threads=args.threads)
    if args.consistency_rounds:
        from .consistency import predict_consistent
        result = predict_consistent(model,payload["state"],payload["questions"],rounds=args.consistency_rounds)
        for answer in result["answers"].values():
            answer.pop("variant_logits",None)
    else:
        try:
            options = {}
            if "domains" in payload:
                if not getattr(model, "accepts_domains", False):
                    raise ValueError("domains require mode routed")
                options["domains"] = payload["domains"]
            result = model.predict(payload["state"], payload["questions"], **options)
            result.pop("logits", None)
        finally:
            if hasattr(model, "close"):
                model.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    parser.error("supply a request.json or --serve")
