#!/bin/bash
# Usage: ./publish.sh <user>@<host>|local [repo-path]
#
# Builds meter-ocr on the target host from this project's GitHub source (not from a local checkout) and
# publishes the result into a custom local pacman repository on that host. "local" builds on this machine
# without ssh. Does NOT install anything -- installing/upgrading is a separate, standard `pacman -S` /
# `pacman -Syu` step once the repo is registered in /etc/pacman.conf (see README.md).
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <user>@<host>|local [repo-path]" >&2
    exit 1
fi

TARGET="$1"
REPO_PATH="${2:-/srv/pacman-repo/meter-ocr}"
REPO_NAME="meter-ocr"
SRC_URL="https://github.com/friedpuppet/meter-ocr.git"

runner=(ssh "$TARGET" bash -s --)
[ "$TARGET" = local ] && runner=(bash -s --)

"${runner[@]}" "$REPO_PATH" "$REPO_NAME" "$SRC_URL" <<'REMOTE'
set -euo pipefail
repo_path="$1"
repo_name="$2"
src_url="$3"
src_dir="$HOME/meter-ocr-src"

if [ -d "$src_dir/.git" ]; then
    git -C "$src_dir" fetch origin
    git -C "$src_dir" reset --hard origin/main
else
    git clone "$src_url" "$src_dir"
fi

cd "$src_dir"
makepkg -f -d  # nothing to compile; pacman installs the runtime deps

pkgfile="$(makepkg --packagelist)"
mkdir -p "$repo_path"
cp -f "$pkgfile" "$repo_path/"
repo-add "$repo_path/$repo_name.db.tar.gz" "$repo_path/$(basename "$pkgfile")"
REMOTE

cat <<MSG

Built and published to $TARGET:$REPO_PATH (repo '$REPO_NAME').

First time only -- register the repo in /etc/pacman.conf:

    [$REPO_NAME]
    SigLevel = Optional TrustAll
    Server = file://$REPO_PATH

then 'sudo pacman -Sy'. After that, install/upgrade is plain pacman:

    sudo pacman -S meter-ocr    # first install
    sudo pacman -Syu            # future upgrades
MSG
