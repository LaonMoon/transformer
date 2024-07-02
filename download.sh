#!/bin/bash
# Most codes used here are referred from below:
# https://github.com/wjcldply/Transformer/blob/main/download.sh
#
SRC=de  # source language (german)
TGT=en  # target language (english)

ROOT="./datasets"  # root directory is (directory of this script)/datasets

ORIG=$ROOT/iwslt17_de.en.orig  # ORIG : original dataset directory
DATA=$ROOT/iwslt17.de.en  # DATA : preprocessed dataset directory
mkdir -p "$ORIG" "$DATA"  # make directory for ORIG & DATA. If parent directories doesn't exists, make them too (-p option)

URL="https://fbk.sharepoint.com/:u:/s/MTUnit/Eb061XjnEwpW2LVP96wNi5gBV580ogirUa1HU9PDiimYhQ?download=1"  # download link
ORIG_FILENAME="2017-01-trnmted.tar"  # specifying the download file's name as desired.

ARCHIVE=$ORIG/$ORIG_FILENAME  # path to the downloaded dataset archive file
# ARCHIVE=$ORIG/2017-01-trnmted.tar  # path to the downloaded dataset archive file
VALID_SET="IWSLT17.TED.dev2010.de-en"  # VALID_SET : name of the validation set


# download and extract data
if [ -f "$ARCHIVE" ]; then
    echo "$ARCHIVE already exists, skipping download"
else
    wget -O "$ORIG/$ORIG_FILENAME" "$URL"  # Download from URL to ORIG directory
    if [ -f "$ARCHIVE" ]; then
        echo "$URL successfully downloaded."
    else
        echo "$URL not successfully downloaded."
        exit 1
    fi
fi


FILE=${ARCHIVE: -4}  # ARCHIVE's last 4 characters; .tar -> save to FILE var
echo "var FILE : $FILE"

if [ -e "$FILE" ]; then  # if FILE format normal file exists
    echo "$FILE already exists, skipping extraction"
else
    tar -C "$ORIG" -xzvf "$ARCHIVE"  # extracts the contents of the dataset archive file to the directory specified by ORIG.
    # -C "$ORIG" -> change the current directory to ORIG directory before extracting
    # -xzvf "$ARCHICE" -> Extracts files from the gzip-compressed archive specified by "$ARCHIVE" to the directory "$ORIG", printing verbose output.
fi

# The extracted folder($ORIG/texts/DeEnItNlRo/DeEnItNlRo/)
# still has nested file(DeEnItNlRo-DeEnItNlRo.tgz) that's not extracted and is still compressed.

EXTRACTED="${ORIG_FILENAME%.*}/texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo"

# Check if the nested directory already exists
if [ -d "$ORIG/$EXTRACTED" ]; then
    echo "Directory $ORIG/${ORIG_FILENAME%.*} already exists, skipping extraction of nested .tgz file"
else
    # Extract the nested .tgz file
    NESTED_TGZ_FILE="$ORIG/$EXTRACTED.tgz"
    tar -xzvf "$NESTED_TGZ_FILE" -C "$(dirname "$NESTED_TGZ_FILE")"
fi


# Pre-Processing Part (Train Data)
echo "pre-processing train data..."
for LANG in "${SRC}" "${TGT}"; do
    # use cat to concatenate and output the contents of the training data file specified by the ${SRC} and ${TGT} variables.
    # ${LANG} is the current language being processed in the loop.

    # use grep -v to filter out lines containing specific XML tags from the data.
    # The -v option in grep is used to invert the match, meaning it will exclude lines matching the specified patterns.

    # use sed to remove specific XML tags (<title>, </title>, <description>, </description>) from the data.

    # redirects the processed data to a file in the ${DATA} directory. The filename includes the source and target languages.
    cat "$ORIG/$EXTRACTED/train.tags.${SRC}-${TGT}.${LANG}" \
        | grep -v '<url>' \
        | grep -v '<talkid>' \
        | grep -v '<keywords>' \
        | grep -v '<speaker>' \
        | grep -v '<reviewer' \
        | grep -v '<translator' \
        | grep -v '<doc' \
        | grep -v '</doc>' \
        | sed -e 's/<title>//g' \
        | sed -e 's/<\/title>//g' \
        | sed -e 's/<description>//g' \
        | sed -e 's/<\/description>//g' \
        | sed 's/^\s*//g' \
        | sed 's/\s*$//g' \
        > "$DATA/train.${SRC}-${TGT}.${LANG}"
done


# Pre-Processing Part (Validation Data)
echo "pre-processing valid data..."
for LANG in "$SRC" "$TGT"; do
    grep '<seg id' "$ORIG/$EXTRACTED/${VALID_SET}.${LANG}.xml" \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
        > "$DATA/valid.${SRC}-${TGT}.${LANG}"
done