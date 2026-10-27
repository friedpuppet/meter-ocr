# Maintainer: friedpuppet <friedpuppet@pm.me>
pkgname=meter-ocr
pkgver=0.1.0
pkgrel=1
pkgdesc="Reads water-meter photos (wheels + serial) with local Ollama models for Home Assistant, only while the host is idle"
arch=('any')
url="https://github.com/friedpuppet/meter-ocr"
license=('MIT')
options=('!debug')
depends=('python' 'python-numpy' 'python-pillow' 'python-opencv' 'ollama')
backup=('etc/meter-ocr/config.json')
install=meter-ocr.install
source=(
    common.py
    locate.py
    crop.py
    read.py
    combine.py
    pipeline.py
    service.py
    meter-ocr.service
    meter-ocr.sysusers
    config.json.example
    LICENSE
)
# Local sources versioned alongside this PKGBUILD, not fetched over the
# network -- nothing to verify a checksum against.
sha256sums=('SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP')

package() {
    # Flat modules importing each other by name: service.py is run as a script from this directory.
    local f
    for f in common locate crop read combine pipeline service; do
        install -Dm644 "$srcdir/$f.py" "$pkgdir/usr/lib/meter-ocr/$f.py"
    done
    install -Dm644 "$srcdir/meter-ocr.service" "$pkgdir/usr/lib/systemd/system/meter-ocr.service"
    install -Dm644 "$srcdir/meter-ocr.sysusers" "$pkgdir/usr/lib/sysusers.d/meter-ocr.conf"
    # Ships as the live default config; backup=() keeps local edits (.pacnew on upgrade).
    install -Dm644 "$srcdir/config.json.example" "$pkgdir/etc/meter-ocr/config.json"
    install -Dm644 "$srcdir/LICENSE" "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
