GITHUB_WORKSPACE="$1"

sudo python3 -m pip install -r "$GITHUB_WORKSPACE"/requirements.txt
sudo python3 "$GITHUB_WORKSPACE"/helper.py