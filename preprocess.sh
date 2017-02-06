#!/usr/bin/env sh

python2 scripts/download.py

CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"
javac -cp $CLASSPATH -Xlint lib/*.java

glove_dir="data/glove"
glove_pre="glove.840B"
glove_dim="300d"
if [ ! -f $glove_dir/$glove_pre.$glove_dim.th ]; then
	th scripts/convert-wordvecs.lua $glove_dir/$glove_pre.$glove_dim.txt \
		$glove_dir/$glove_pre.vocab $glove_dir/$glove_pre.$glove_dim.th
fi

cat $glove_dir/$glove_pre.$glove_dim.txt | cut -d ' ' -f 2- > $glove_dir/$glove_pre.$glove_dim.yee

python3 scripts/preprocess-glove.py $glove_dir/$glove_pre.$glove_dim.yee $glove_dir/$glove_pre.$glove_dim.emb
python3 scripts/preprocess-toefl.py
python2.7 scripts/parse_query.py

for frac in 0.1 0.2 0.3 0.4 0.5 1.0 ;do
	python3 scripts/prune.py $frac
	python3 scripts/dependency_parse.py $frac
done

mkdir -p trained_models
rm .trash
rm .trash2
