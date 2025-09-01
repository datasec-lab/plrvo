wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf datasets.tar

mkdir -p auditing/original/GLUE-SST-2
patch original/GLUE-SST-2/train.tsv auditing/patch/GLUE-SST-2/train.patch -o auditing/original/GLUE-SST-2/train.tsv
cp original/GLUE-SST-2/dev.tsv auditing/original/GLUE-SST-2/dev.tsv
cp original/GLUE-SST-2/test.tsv auditing/original/GLUE-SST-2/test.tsv

mkdir -p auditing/original/QNLI
patch original/QNLI/train.tsv auditing/patch/QNLI/train.patch -o auditing/original/QNLI/train.tsv
cp original/QNLI/dev.tsv auditing/original/QNLI/dev.tsv
cp original/QNLI/test.tsv auditing/original/QNLI/test.tsv

