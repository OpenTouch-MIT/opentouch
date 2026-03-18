#!/bin/bash
set -euo pipefail

DATA_DIR="data"
if ! mkdir -p "$DATA_DIR"; then
    echo "Error: Could not create directory $DATA_DIR" >&2
    exit 1
fi
cd "$DATA_DIR"

IDS=(
    "1EjMOzs45devBo0TqhuhZTT_Ll7HZ1lrW"
    "1fAmmieSr0yFm7ldhW7Smld7jUxBCw8fu"
    "1cUhgYbredkIRswanUiM5uDixiFLq4WCC"
    "1PCzWMJxtbD2HJLCl2WFzOIB-5RN3X81G"
    "1jFlYmCFb6GldbPJ-zLzJSCY-BKipdjPE"
    "1reSqa8v8RaY2kZXLw0_g7Amvq7lJl6Cu"
    "1atXpcctoHs4dbXhyAAO9EY88D2f1JYfT"
    "1Z3b-I6BMPgNlpiKw8gISkUi3VULUtLFN"
    "1u-6WGn3eMQJe3eh6lCFahlIcEVmkULna"
    "17wF0aBIH6RRtRGRaXeiI-Y4Lh5bnDFBL"
    "1KICpqtfmbnKhgHi-CIR9XAp24TE1945M"
    "1vkl6wat_dgF5NQs9QVDfCyJGyjEjd2FW"
    "1BbKU5vSH-wOrCnOjRWNe3H7niJP_uJrb"
    "1GCX4mAgCvOvmIQ0uXotqpoNdXgYzp4ki"
    "1rxsLWGw_diPvRnALxOYakCIweG90O28I"
    "1zAYfcMt2hqcG1bPtCOAkWu0zsd6lfvrX"
    "1tQh21z8KRxYHsh69dW6VcSw5Wux67R6_"
    "1jeA1bEit-tDQpfwt3NmTeC8iwM6I1qiE"
    "1UT5htydKCfBCO57On-mRJRz7mSi57K4u"
    "1h9Bl8CTGJWvU2XPr93fptBTpB2cwYgwq"
    "1SAbxWQZDEyTZ-ESVi9G5bxEc7ov-EO28"
    "1jKyVNsi7fsofSho_xoRi0Kgqem4zrk5F"
    "11LQ28c6jPhNfiu9fPDu5diruNUCa0bGM"
    "1dwlVYtBfyNUHg7Qxnxa_iYBYPCcn9VeX"
    "1X4-MS7Qodhtmn6zcY9a5cMq02eDLvOJq"
    "1VAKXJPO4j_40hpqslNJ4_WbgWfKaGLQC"
    "1cM-816vcCnkgWVIGXZrR1o8TPsDvRVCZ"  # labels
)

for ID in "${IDS[@]}"; do
    gdown "$ID"
done

echo "All datasets downloaded successfully."