
URL=$1
sudo apt-get update
sudo apt-get install aria2

aria2c -o webimage.sh -x 16 "$URL"

chmod +x webimage.sh
sudo ./webimage.sh -b -p /intel/oneapi/intelpython -k
installer_exit_code=$?
exit $installer_exit_code