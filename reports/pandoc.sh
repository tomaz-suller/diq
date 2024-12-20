PANDOC=pandoc
FLAGS="\
    --from=markdown+rebase_relative_paths+raw_tex \
    --to=pdf \
    --metadata-file=metadata.yaml \
    --bibliography=bibliography.bib \
    --citeproc\
"

sh -c "${PANDOC} ${FLAGS} --output=${2} ${1}"
