#!/bin/bash
echo "[+] Creating dataset file named dataset.json ..."
touch dataset.json
echo '{ "data" : [' >> dataset.json
for i in $(curl https://padawan.s3.eurecom.fr/static/data/dataset_sha256.txt | awk '{print $1}')
do
	echo "[+] Requesting $i ..."
	curl https://padawan.s3.eurecom.fr/static/dataset/"$i".json >> dataset.json
	if [ $i != "bd92ce74844b1ddfdd1b61eac86abe7140d38eedf9c1b06fb7fbf446f6830391" ]
	then
		echo "," >> dataset.json
	fi
done
echo "] }" >> dataset.json
