cd "$SRC"/timeparser
pip3 install .

# Build fuzzers in $OUT
for fuzzer in $(find fuzzing -name '*_fuzzer.py');do
  compile_python_fuzzer "$fuzzer" --add-binary="timeparser/data:timeparser/data"
done
zip -q $OUT/timeparser_fuzzer_seed_corpus.zip $SRC/corpus/*
