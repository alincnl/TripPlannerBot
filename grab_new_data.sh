#!/bin/bash

RESULTS_FOLDER="test_data_grab"
POSTPROCESS_RESULTS_FOLDER="${RESULTS_FOLDER}/processed"


if [ ! -d "$RESULTS_FOLDER" ]; then
    echo "====== Making folder: '${RESULTS_FOLDER}' ======"
    mkdir "${RESULTS_FOLDER}"
fi

echo "====== Call 'cafes-parser' ======"
cafes-parser "${RESULTS_FOLDER}/cafes.csv"

echo "====== Call 'ent-parser' ======"
ent-parser "${RESULTS_FOLDER}/ent.csv"

echo "====== Call 'hotels-parser' ======"
hotels-parser "${RESULTS_FOLDER}/hotels.csv"

if [ ! -d "$POSTPROCESS_RESULTS_FOLDER" ]; then
    echo "====== Making folder: '${POSTPROCESS_RESULTS_FOLDER}' ======"
    mkdir "${POSTPROCESS_RESULTS_FOLDER}"
fi

echo "====== Call 'preprocess-data' ======"
preprocess-data \
--hotels_csv "${RESULTS_FOLDER}/hotels.csv" \
--nsk_cafes_csv "${RESULTS_FOLDER}/cafes.csv" \
--nso_cafes_csv ./src/datasets/compiled_cafes_nso.csv \
--ent_csv "${RESULTS_FOLDER}/ent.csv" \
--nso_nature_csv ./src/datasets/dataset_nature.csv \
--output_folder "${POSTPROCESS_RESULTS_FOLDER}"

echo "====== Call 'get-coords' ======"
get-coords "[${POSTPROCESS_RESULTS_FOLDER}/production_hotels.csv, \
${POSTPROCESS_RESULTS_FOLDER}/production_cafes.csv, \
${POSTPROCESS_RESULTS_FOLDER}/production_ent.csv]"

echo "====== Call 'get-districts' ======"
get-distircts \
--hotels_csv ${POSTPROCESS_RESULTS_FOLDER}/production_hotels.csv \
--ent_csv ${POSTPROCESS_RESULTS_FOLDER}/production_ent.csv \
--cafes_csv ${POSTPROCESS_RESULTS_FOLDER}/production_cafes.csv