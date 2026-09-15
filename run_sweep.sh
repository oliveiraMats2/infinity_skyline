#!/usr/bin/env bash
#
# Sweep end to end sobre configs/experiments/<prefixo>_*.yaml.
#
# Isto e orquestracao pura: todo passo e o CLI que ja existe (main.py convert e
# main.py panorama). Nada do pipeline e reimplementado aqui, e cada run escreve o
# proprio results/<run_id>/run.log, que e de onde este script le o resumo.
#
#   ./run_sweep.sh                     # cena padrao, data/high_bright
#   ./run_sweep.sh data/low_bright     # outra cena
#   PYTHON=python ./run_sweep.sh       # outro interpretador
#
# Uma variante que falha nao interrompe as outras: proj_planar, por exemplo, deve
# mesmo abortar nestas cenas, e isso e o resultado dela.
#
set -uo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-.venv/bin/python}
SOURCE=${1:-data/high_bright}
PREFIXES=(det match strategy ransac graph proj ref focal wave)

echo "==> convert  $SOURCE"
"$PYTHON" main.py convert --config configs/default.yaml --source "$SOURCE" || exit 1

printf '\n%-22s %-8s %-13s %s\n' CONFIG STATUS CANVAS ORDEM
printf '%.0s-' {1..100}; echo

total=0
failed=0
for prefix in "${PREFIXES[@]}"; do
    for config in configs/experiments/"${prefix}"_*.yaml; do
        name=$(basename "$config" .yaml)
        total=$((total + 1))
        if "$PYTHON" main.py panorama --config "$config" >/dev/null 2>&1; then
            log="results/$name/run.log"
            canvas=$(grep -o 'canvas: [0-9]*x[0-9]*' "$log" | tail -1 | cut -d' ' -f2)
            order=$(grep -o 'inferred order.*' "$log" | tail -1 | sed 's/.*: //; s/IMG_//g; s/ -> /,/g')
            printf '%-22s %-8s %-13s %s\n' "$name" ok "$canvas" "$order"
        else
            reason=$(grep -oE '(ValueError|BadParameter|Error): .*' "results/$name/run.log" 2>/dev/null | tail -1)
            printf '%-22s %-8s %-13s %s\n' "$name" FALHOU - "${reason:-veja results/$name/run.log}"
            failed=$((failed + 1))
        fi
    done
done

echo
echo "$((total - failed)) de $total concluiram."
echo "panoramas: results/<config>/panorama/panorama.png"
echo "comparar: abra dois panoramas do mesmo prefixo lado a lado."
