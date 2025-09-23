set -euo pipefail

REPO_URL="https://github.com/luisrui/Modality-Interference-in-MLLMs.git"
REPO_DIR="Modality-Interference"
REQ_FILE="requirements_llava.txt"

REPO_ID="luisrui/Modality-Interference-in-MLLMs-DATA"
REPO_TYPE="dataset"
DEST_DIR="${1:-./data}"

need_cmd git
need_cmd python3 || need_cmd python
need_cmd pip || need_cmd python3

if [[ ! -d "$REPO_DIR" ]]; then
  log "Cloning repo into ${REPO_DIR} ..."
  git clone "$REPO_URL" "$REPO_DIR"
else
  log "Repo ${REPO_DIR} already exists. Pulling latest..."
  (cd "$REPO_DIR" && git pull --ff-only)
fi

cd "$REPO_DIR"

if [[ -f "$REQ_FILE" ]]; then
  log "Installing Python deps from ${REQ_FILE} ..."
  pip install -r "$REQ_FILE"
else
  die "Requirements file ${REQ_FILE} not found."
fi

log "Target HF dataset repo: ${REPO_ID}"
log "Output dir           : ${DEST_DIR}"
mkdir -p "${DEST_DIR}"

# 0) Ensure huggingface-cli is available
if ! command -v huggingface-cli >/dev/null 2>&1; then
  log "==> huggingface-cli not found. Installing huggingface_hub..."
  if command -v pip >/dev/null 2>&1; then
    pip install -U huggingface_hub
  else
    python3 -m pip install -U huggingface_hub
  fi
fi

# 1) Download all files in the dataset repo to DEST_DIR
#    --local-dir-use-symlinks False to get real files instead of symlinks
log "==> Downloading files from Hugging Face (this may take a while)..."
huggingface-cli download "${REPO_ID}" \
  --repo-type "${REPO_TYPE}" \
  --local-dir "${DEST_DIR}" \
  --local-dir-use-symlinks False \
  --include "*" \
  --exclude ".gitattributes" ".git/*"

log "==> Download completed."

# 2) Reassemble split archives (LLaVA-Instruct-665K.tar.gz.part_aa, ab, ...)
log "==> Checking for split archives to reassemble..."
shopt -s nullglob
parts=( "${DEST_DIR}/LLaVA-Instruct-665K.tar.gz.part_"* )
if (( ${#parts[@]} > 0 )); then
  # Sort parts to ensure aa, ab, ac ... order
  IFS=$'\n' parts_sorted=($(printf "%s\n" "${parts[@]}" | sort))
  unset IFS
  target="${DEST_DIR}/LLaVA-Instruct-665K.tar.gz"
  log "    Found ${#parts_sorted[@]} parts. Reassembling -> $(basename "${target}")"
  cat "${parts_sorted[@]}" > "${target}"
  log "    Reassembled."
else
  log "    No split parts found."
fi

# 3) Extract all .tar.gz archives (including the reassembled one)
log "==> Extracting *.tar.gz archives..."
archives=( "${DEST_DIR}"/*.tar.gz )
if (( ${#archives[@]} == 0 )); then
  log "    No .tar.gz archives found to extract."
else
  for f in "${archives[@]}"; do
    name="$(basename "${f}")"
    log "    Extracting ${name} ..."
    # Extract into DEST_DIR (archives usually contain their own top-level folder)
    tar -xzf "${f}" -C "${DEST_DIR}"
    log "    OK: ${name}"
  done
fi

# 4) Cleanup: remove archives and split parts after successful extraction
log "==> Cleaning up archives and split parts..."
# Remove .tar.gz (only if extraction reached here without error)
for f in "${DEST_DIR}"/*.tar.gz; do
  [ -e "$f" ] && rm -f "$f"
done
# Remove split parts
for f in "${DEST_DIR}"/LLaVA-Instruct-665K.tar.gz.part_*; do
  [ -e "$f" ] && rm -f "$f"
done

log "==> All done!"
log "Contents are in: ${DEST_DIR}"

# 5) Run 3 inference jobs 
set +e
log "Running inference: llava-v1.6-34b-hf ..."
bash zs_inference.sh \
  --model_name llava-v1.6-34b-hf \
  --checkpoint_path llava-hf/llava-v1.6-34b-hf \
  --batch_size 4 \
  --tag origin \
  --all

log "Running inference: llava-next-72b-hf ..."
bash zs_inference.sh \
  --model_name llava-next-72b-hf \
  --checkpoint_path llava-hf/llava-next-72b-hf \
  --batch_size 4 \
  --tag origin \
  --all

log "Running inference: llava-next-110b-hf ..."
bash zs_inference.sh \
  --model_name llava-next-110b-hf \
  --checkpoint_path llava-hf/llava-next-110b-hf \
  --batch_size 4 \
  --tag origin \
  --all
set -e

log "All done!"