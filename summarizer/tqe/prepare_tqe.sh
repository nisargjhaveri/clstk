#!/bin/bash

# TODO: Don't use paths here directly
TERCOM_JAR=~/Libraries/tercom-0.7.25/tercom.7.25.jar

function usage() {
    echo "Usage: prepare_tqe.sh [-h] [--prepared] SRC_FILE MT_FILE REFS_FILE OUT_DIR MODEL_NAME" >&2
    echo "The files have sentence id and tokens per line seperated by tab" >&2
    exit 1
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --prepared)
            PREPARED=YES
            shift
            ;;
        -*)
            echo "Unknown argument '$1'"
            usage
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

[ $# -eq 5 ] || usage

SRC_FILE=$1
MT_FILE=$2
REFS_FILE=$3
OUT_DIR=$4
OUT_FILE="tqe.$5"

mkdir -p "$OUT_DIR"

echo "==> Preparing files for HTER <=="
if [ "$PREPARED" = "YES" ]; then
    cat "$SRC_FILE" > "$OUT_DIR/$OUT_FILE.src"
    cat "$MT_FILE" > "$OUT_DIR/$OUT_FILE.mt"
    cat "$REFS_FILE" > "$OUT_DIR/$OUT_FILE.ref"
else
    sed -r "s/([^\t]*)\t(.*)/\2\t(\1)/g" "$SRC_FILE" > "$OUT_DIR/$OUT_FILE.src"
    sed -r "s/([^\t]*)\t(.*)/\2\t(\1)/g" "$MT_FILE" > "$OUT_DIR/$OUT_FILE.mt"
    sed -r "s/([^\t]*)\t(.*)/\2\t(\1)/g" "$REFS_FILE" > "$OUT_DIR/$OUT_FILE.ref"
fi

echo "==> Computing HTER scores <=="
java -jar "$TERCOM_JAR" -h "$OUT_DIR/$OUT_FILE.mt" -r "$OUT_DIR/$OUT_FILE.ref" -o ter -n "$OUT_DIR/$OUT_FILE" > /dev/null
tail -n +3 "$OUT_DIR/$OUT_FILE.ter" | awk '{print $4}' > "$OUT_DIR/$OUT_FILE.hter"
rm "$OUT_DIR/$OUT_FILE.ter"

echo "==> Preparing files for BLEU <=="
sed -i -r "s/(.*)\t(.*)/\1/g" "$OUT_DIR/$OUT_FILE.src"
sed -i -r "s/(.*)\t(.*)/\1/g" "$OUT_DIR/$OUT_FILE.mt"
sed -i -r "s/(.*)\t(.*)/\1/g" "$OUT_DIR/$OUT_FILE.ref"

echo "==> Computing BLEU scores <=="
python > "$OUT_DIR/$OUT_FILE.bleu" <<END
from nltk.translate import bleu_score

with open("$OUT_DIR/$OUT_FILE.mt") as hyp_file:
    with open("$OUT_DIR/$OUT_FILE.ref") as ref_file:
        # print bleu_score.corpus_bleu(
        #     [[_.split()] for _ in list(ref_file)],
        #     [_.split() for _ in list(hyp_file)]
        # )

        for ref, hyp in zip([_.split() for _ in list(ref_file)], [_.split() for _ in list(hyp_file)]):
            print bleu_score.sentence_bleu([ref], hyp, smoothing_function=bleu_score.SmoothingFunction().method1)
END

echo "==> Finalizing <=="
