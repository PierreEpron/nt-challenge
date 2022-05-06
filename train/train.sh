#! /bin/bash
#
DATAPATH="train/$1/data/"
echo "data will be look at path $DATAPATH"
CONFIGPATH="train/$1/$2/config.yml"
echo "data will be look at path $CONFIGPATH"
DOMAINPATH="train/$1/$2/domain.yml"
echo "data will be look at path $DOMAINPATH"
MODELNAME="$1-$2"
echo "$MODELNAME will be use as name for model"
TESTSPATH="train/$1/tests/"
echo "tests data will be look at path $TESTSPATH"
RESULTSPATH="train/$1/$2/results/"
echo "tests result will be store at path $RESULTSPATH"
USERTESTSPATH="train/$1/user_tests/"
echo "data will be look at path $USERTESTSPATH"
USERRESULTSPATH="train/$1/$2/user_results/"
echo "user tests result will be store at path $USERRESULTSPATH"
# 
python -m rasa train --data $DATAPATH --config $CONFIGPATH --domain $DOMAINPATH --fixed-model-name $MODELNAME
python -m rasa test nlu --model "models/$MODELNAME.tar.gz" --out $RESULTSPATH --nlu $TESTSPATH --config $CONFIGPATH --domain $DOMAINPATH
python -m rasa test nlu --model "models/$MODELNAME.tar.gz" --out $USERRESULTSPATH --nlu $USERTESTSPATH --config $CONFIGPATH --domain $DOMAINPATH
#
$SHELL