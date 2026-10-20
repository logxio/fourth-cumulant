#!/bin/bash
# Cloud box setup for end-to-end study: Python 3.13 venv with CPU torch, flopscope 0.12.3, whestbench 0.16.x.
# usage (on the box, inside ${WORKDIR:-$PWD} unpacked from e2e_pkg.tgz): bash lab/e2e_s1/remote_setup.sh
set -euo pipefail
cd "${WORKDIR:-$PWD}"
if ! command -v curl >/dev/null || [ ! -x /usr/bin/time ]; then apt-get update -qq && apt-get install -y -qq curl ca-certificates time >/dev/null; fi
if [ ! -x "$HOME/.local/bin/uv" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13 >/dev/null
[ -d venv ] || uv venv -q -p 3.13 venv
uv pip install -q -p venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -q -p venv/bin/python "flopscope>=0.12.3,<0.13" "whestbench>=0.16.1,<0.17" einops jaxtyping tqdm psutil scipy numpy joblib pandas matplotlib tabulate blobfile opt-einsum
venv/bin/python -c "import torch, flopscope, numpy; print('torch', torch.__version__, 'flopscope', flopscope.__version__, 'numpy', numpy.__version__, 'threads', torch.get_num_threads()); numpy.show_config()" 2>&1 | head -40
nproc; free -g | head -2
