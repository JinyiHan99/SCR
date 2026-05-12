AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)

MODELS=(
    '/SCR/ckp/SCR-SFT'
    '/SCR/ckp/SCR-Stage1'
    '/SCR/ckp/SCR-Stage2'
)
BASE_DIR='/SCR/logs/results'
LOG_DIR="${BASE_DIR}/logs"
OUT_DIR="${BASE_DIR}/results"
mkdir -p "$LOG_DIR" "$OUT_DIR"

PROMPTS=(
    'SCR_prompt'
    'SCR_prompt'
    'SCR_prompt'
)

TASKS=(
  "aime25"
  "math500"
  "aime24"
  "olypaid"
  "amc"
  "arc"
  "gpqa"
  "mmlu_pro"
)

BATCH_SIZE=250
NUM_WORKERS=0
USE_CHAT_TEMPLATE=1
DTYPE="bfloat16"
ENFORCE_EAGER=0



JOB_PIDS=()
GPU_COUNT=${#AVAILABLE_GPUS[@]}

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  GPU="${AVAILABLE_GPUS[$i % GPU_COUNT]}"  
  prompt="${PROMPTS[$i]}"

  SAFE_NAME="${MODEL//\//_}"
  TS="$(date +%Y%m%d-%H%M%S)"
  LOG_FILE="${LOG_DIR}/${TS}-${SAFE_NAME}.log"
  OUT_FILE="${OUT_DIR}/${TS}-${SAFE_NAME}.jsonl"

  echo ">>> Launch model: ${MODEL} on GPU ${GPU}"
  echo "    Log: ${LOG_FILE}"
  echo "    Out: ${OUT_FILE}"

  (
    export CUDA_VISIBLE_DEVICES="${GPU}"

    ARGS=( ./infer/infer_all_v2.py
      --model_path "${MODEL}"
      --tasks "${TASKS[@]}"
      --batch_size "${BATCH_SIZE}"
      --output "${OUT_FILE}"
      --dtype "${DTYPE}"
      --chat_system_prompt "${prompt}"
    )
    [[ "${USE_CHAT_TEMPLATE}" == "1" ]] && ARGS+=( --use_chat_template )

    echo ">>> Running command:" > "${LOG_FILE}"
    printf ' %q' "${ARGS[@]}" >> "${LOG_FILE}"
    echo -e "\n>>> End of command info\n" >> "${LOG_FILE}"

    python -u "${ARGS[@]}" >> "${LOG_FILE}" 2>&1 \
      || echo "!!! ${MODEL} failed. Check log: ${LOG_FILE}"
  ) & 

  JOB_PIDS+=($!)
done
echo ">>> All jobs launched. Waiting for completion..."
wait "${JOB_PIDS[@]}"

echo ">>>finished."