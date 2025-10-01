#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

cd data/assets/flame/

# username and password input
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
# read -p "Username (FLAME):" username
# read -p "Password (FLAME):" password
username=$(urle $FLAME_USERNAME)
password=$(urle $FLAME_PWD)

echo -e "\nDownloading FLAME..."
mkdir -p FLAME2023/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1' -O './FLAME2023.zip' --no-check-certificate --continue
unzip FLAME2023.zip -d ./
rm -rf FLAME2023.zip

mv FLAME2023/flame2023_no_jaw.pkl ./
rm -rf FLAME2023

# Fix the FLAME pickle file for compatibility issues
cd ../../../
python scripts/fixes/fix_flame_pickle.py --pickle_path data/assets/flame/flame2023_no_jaw.pkl

echo -e "\nInstallation has finished. If there were any error messages, follow README to download and unzip FLAME2023 manually!"