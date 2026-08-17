r"""Check a machine is ready to run the Alpamayo2-Super planner, before the download.

The 34B checkpoint is ~68 GB and the repo is gated, so the expensive failures
are the ones worth catching first: an unauthenticated Hugging Face session that
401s partway through a batch, a missing flash-attn, or a GPU that cannot hold
the weights. Run this on the target host:

    python scripts/alpamayo2_preflight.py

It only reports; it changes nothing and downloads nothing. Exit code is 0 when
every REQUIRED check passes.

Deliberately standalone -- it runs under the planner's interpreter, not the
OptiCarVis venv, so it imports nothing from this repo.
"""

import importlib
import os
import platform
import shutil
import sys


# Floors come from the alpamayo2 package's own pyproject.
REQUIRED_TORCH = (2, 8)
REQUIRED_TRANSFORMERS = (4, 57, 1)

# BF16 weights alone; leave headroom for activations and the KV cache.
WEIGHTS_GIB = 68


def version_tuple(text):
    parts = []

    for chunk in str(text).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)

    return tuple(parts)


class Report(object):
    def __init__(self):
        self.failed = 0
        self.warned = 0

    def ok(self, label, detail=""):
        print("  PASS  %-22s %s" % (label, detail))

    def warn(self, label, detail=""):
        self.warned += 1
        print("  WARN  %-22s %s" % (label, detail))

    def fail(self, label, detail=""):
        self.failed += 1
        print("  FAIL  %-22s %s" % (label, detail))


def check_platform(report):
    if platform.system() == "Linux":
        report.ok("os", platform.platform())
    else:
        report.fail("os", "%s -- the planner is Linux + CUDA only" % platform.system())

    report.ok("python", sys.version.split()[0])


def check_package(report, module, label, floor=None):
    try:
        mod = importlib.import_module(module)
    except Exception as error:
        report.fail(label, "not importable (%s)" % type(error).__name__)
        return None

    version = getattr(mod, "__version__", "unknown")

    if floor and version != "unknown" and version_tuple(version) < floor:
        report.fail(label, "%s < required %s" % (version, ".".join(str(v) for v in floor)))
    else:
        report.ok(label, str(version))

    return mod


def check_gpu(report, torch):
    if torch is None:
        return

    if not torch.cuda.is_available():
        report.fail("cuda", "torch.cuda.is_available() is False")
        return

    report.ok("cuda", "toolkit %s, %d device(s)" % (torch.version.cuda, torch.cuda.device_count()))

    total = 0.0
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / (1024 ** 3)
        total += gib
        print("        device %d: %s, %.1f GiB" % (index, props.name, gib))

    if total >= WEIGHTS_GIB * 1.15:
        report.ok("gpu memory", "%.1f GiB total" % total)
    elif total >= WEIGHTS_GIB:
        report.warn("gpu memory", "%.1f GiB is above the ~%d GiB of weights but leaves little headroom"
                    % (total, WEIGHTS_GIB))
    else:
        report.fail("gpu memory", "%.1f GiB cannot hold ~%d GiB of BF16 weights"
                    % (total, WEIGHTS_GIB))


def check_hf_auth(report):
    """The model repo is gated; an unauthenticated run 401s mid-batch."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        report.ok("hf auth", "token in environment")
        return

    try:
        from huggingface_hub import HfFolder

        if HfFolder.get_token():
            report.ok("hf auth", "cached token (hf auth login)")
            return
    except Exception:
        pass

    report.fail("hf auth", "no token -- nvidia/Alpamayo2-Super is GATED and will 401")


def check_alpamayo_package(report):
    try:
        importlib.import_module("alpamayo2_super.models.alpamayo2_super")
        report.ok("alpamayo2_super", "importable")
    except Exception as error:
        report.fail(
            "alpamayo2_super",
            "not importable (%s) -- clone NVlabs/alpamayo2 and pip install it"
            % type(error).__name__,
        )


def check_wrapper(report):
    wrapper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpamayo2_super_wrapper.py")

    if os.path.isfile(wrapper):
        report.ok("wrapper", wrapper)
    else:
        report.fail("wrapper", "alpamayo2_super_wrapper.py not found beside this script")


def check_tooling(report):
    for name in ("ffmpeg", "nvcc"):
        path = shutil.which(name)

        if path:
            report.ok(name, path)
        elif name == "ffmpeg":
            report.warn(name, "not on PATH -- renders keep the mp4v master instead of H.264")
        else:
            report.warn(name, "not on PATH -- flash-attn compiles at install and needs CUDA Toolkit 12.x+")


def main():
    print("")
    print("Alpamayo2-Super preflight")
    print("=========================")

    report = Report()

    print("\nplatform")
    check_platform(report)

    print("\npackages")
    torch = check_package(report, "torch", "torch", REQUIRED_TORCH)
    check_package(report, "transformers", "transformers", REQUIRED_TRANSFORMERS)
    check_package(report, "cv2", "opencv")
    check_package(report, "flash_attn", "flash-attn")
    check_alpamayo_package(report)

    print("\ngpu")
    check_gpu(report, torch)

    print("\naccess")
    check_hf_auth(report)

    print("\nlocal")
    check_wrapper(report)
    check_tooling(report)

    print("")

    if report.failed:
        print("%d check(s) failed, %d warning(s) -- fix the failures before a batch run."
              % (report.failed, report.warned))
        print("Plumbing can still be exercised without the model:")
        print("    python scripts/alpamayo2_super_wrapper.py --clips m.csv "
              "--output-dir out --config c.json --self-test")
        return 1

    print("All required checks passed (%d warning(s))." % report.warned)
    print("Next: run the wrapper with --self-test, then one real clip, before the full batch.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
