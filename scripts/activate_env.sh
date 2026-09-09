# Source this file from WSL: source scripts/activate_env.sh
# No installation, downloads, or checks run during activation.
if [ ! -f /root/.venvs/foliorecall_env/bin/activate ]; then
    printf '%s\n' 'Missing /root/.venvs/foliorecall_env; see README.' >&2
    return 1
fi
source /root/.venvs/foliorecall_env/bin/activate
export HF_HOME=/mnt/d/foliorecall_cache/huggingface
export TMPDIR=/mnt/d/foliorecall_cache/tmp
