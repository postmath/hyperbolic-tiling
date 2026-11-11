#!/usr/bin/env bash
# Renders the icons in icons/ from one view. Needs ImageMagick 7. Rerun and
# commit the output after changing the view or the crops.
set -euo pipefail
cd "$(dirname "$0")/.."

# {5,6} mod 541, panned so that the central tile's color stands out from its
# neighbours'. Lines are off: at icon sizes the color changes mark the edges.
VIEW='v1,p=5,q=6,ell=541,mu=0.5,chart=disk,size=2048x2048,m=1.0310163936424e0:-1.3041428917881284e-1:-2.1444560414558472e-1:-2.3344134883633386e-1:1.8442020458265704e-1:1.0101901066077703e0:9.219512250034129e-2:-9.914620895242067e-1:1.5969616667017086e-1,frame=255:330:205:71:231:109:265:405:74,line=0,dot=0,colors=1'
C=1024     # the center, half of size=
R=1016     # the disk's radius: C less the 8 px margin

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cargo run --release --quiet -- --view "$VIEW" --export "$tmp/view.png"
mkdir -p icons

# round <px> <fraction of R>: the middle of the disk, transparent outside a
# circle. The smaller the icon, the less of the disk it keeps, as the small
# tiles near the rim become noise.
round() {
    local r
    r=$(awk "BEGIN { printf \"%d\", $R * $2 }")
    magick "$tmp/view.png" \
        \( +clone -fill black -colorize 100 -fill white -draw "circle $C,$C $C,$((C - r))" \) \
        -alpha off -compose CopyOpacity -composite \
        -crop "$((2 * r))x$((2 * r))+$((C - r))+$((C - r))" +repage \
        -filter Lanczos -resize "$1x$1" -strip "icons/favicon-$1.png"
}
round 16 0.65
round 32 0.8
round 48 0.8

# iOS fills transparency with black, so the touch icon keeps the background.
magick "$tmp/view.png" -filter Lanczos -resize 180x180 -alpha off -strip \
    icons/apple-touch-icon.png
