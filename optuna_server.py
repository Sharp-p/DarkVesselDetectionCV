"""Local Optuna gRPC storage server, optionally accompanied by its web dashboard."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys


def make_storage(url):
    from optuna.storages import RDBStorage
    from optuna.storages import get_storage
    kwargs = {}
    if url.startswith("sqlite:///"):
        filename = url[len("sqlite:///"):]
        if filename != ":memory:":
            Path(filename).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        kwargs["engine_kwargs"] = {"connect_args": {"timeout": 60}}
    # get_storage wraps RDBStorage with the cache required by the gRPC docs.
    return get_storage(RDBStorage(url, **kwargs))


def client_storage(url, grpc_host=None, grpc_port=13000):
    if grpc_host:
        from optuna.storages import GrpcStorageProxy
        proxy = GrpcStorageProxy(host=grpc_host, port=grpc_port)
        proxy.wait_server_ready(timeout=15)
        return proxy
    return make_storage(url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default="sqlite:///tuning_runs/optuna.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13000)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard-port", type=int, default=8080)
    args = parser.parse_args()
    from optuna.storages import run_grpc_proxy_server
    storage = make_storage(args.storage)
    dashboard = None
    try:
        if args.dashboard:
            dashboard_cli = Path(sys.executable).parent / "optuna-dashboard"
            dashboard = subprocess.Popen([str(dashboard_cli), args.storage,
                                          "--host", args.host, "--port", str(args.dashboard_port)])
            print(f"Dashboard: http://{args.host}:{args.dashboard_port}", flush=True)
        print(f"Storage server: {args.host}:{args.port}. Training runs in tuning.py workers.", flush=True)
        # Serialize SQLite operations. For multiple machines use PostgreSQL instead.
        with ThreadPoolExecutor(max_workers=1 if args.storage.startswith("sqlite:") else 10) as pool:
            run_grpc_proxy_server(storage, host=args.host, port=args.port, thread_pool=pool)
    except KeyboardInterrupt:
        pass
    finally:
        if dashboard is not None:
            dashboard.terminate()
            try:
                dashboard.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard.kill()
                dashboard.wait()


if __name__ == "__main__":
    main()
