# meter-ocr

An Arch Linux package (`meter-ocr`) that reads household water-meter photos for Home Assistant with local Ollama
models: the serial number and the 8 number wheels (5 black, 3 red). It runs on a PC with a GPU, not on the Home
Assistant host, and **reads only while that PC has been idle**. Home Assistant decides when to ask. This service only
answers, and says "busy" when the PC is in use, for example while you play a game.

Photos never leave the host: the service talks to Ollama on `localhost` only.

## What it does with a photo

1. **Locate.** `qwen3-vl:8b-instruct` boxes the dial, the counter window, the «NOVATOR» logo and the serial. It
   locates them and reads nothing.
2. **Level and crop, in code (OpenCV).**
   - Coarse rotation: "up" is from the dial's centre to the logo, and from the serial to the centre.
   - Fine rotation: the peak of the long straight edges inside the dial.
   - Counter box: the window's frame, as the pair of horizontal edges with the frame's aspect (4.5 : 1) and the
     dark wheels between them.
   - Shear removal: makes the vertical structure upright.
3. **Read.** `glm-ocr` and `qwen3-vl` each read the window. Widening crops are tried until one gives exactly 8
   digits. `glm-ocr` reads the serial.
4. **Agree.** A wheel keeps its digit only where both models read the same one; otherwise it is `?`. A reading
   that isn't 8 digits, or whose first digit isn't `0`, leaves all black wheels `?`.

Result per photo: `{"file", "serial", "black": "00123", "red": "4?6", "note"}`. Matching the serial to a meter,
voting over several photos and range checks are left to the caller (a Home Assistant script).

## HTTP API

Every request needs `Authorization: Bearer <token>`.

| Request | Answer |
|---|---|
| `GET /status` | `{"ready": bool, "reason": "...", "idle_min": 7.5, "job": null}` |
| `POST /batch` (multipart, one or more photos) | `202 {"id"}`. Not idle: `503 {"reason"}`. A batch is running: `409` |
| `GET /batch/<id>` | `{"state": "running"\|"done"\|"failed", "results": [...], "seconds"}` |

**Ready** means GPU (`gpu_busy_percent`) and CPU have both been under their limits (30 %) for the last 10 minutes,
sampled every 10 s.
- Samples taken during the service's own batch don't count.
- A gap in sampling (sleep, suspend) restarts the window.

A started batch always finishes. The models are loaded with a short `keep_alive`, so VRAM is freed soon after.
A batch of 4 photos takes about 30 s on a Radeon RX 9070 XT.

## Files

- `service.py`: the HTTP service (stdlib server).
- `pipeline.py`: the batch pipeline in memory.
- `locate.py`, `crop.py`, `read.py`, `combine.py`: the steps. Each also runs on its own, writing to
  `~/.cache/meter-ocr` (`$METER_OCR_CACHE`), for tuning against your own photos.
- `common.py`: shared helpers.
- `/etc/meter-ocr/config.json`: listen address/port, `gpu_max`, `cpu_max`, `idle_minutes`, `sample_seconds`,
  `keep_alive`. It is in `backup=()`, so local edits survive upgrades.
- `/var/lib/meter-ocr/token`: generated on first start (mode 600, user `meter-ocr`). Give it to Home Assistant.
- `meter-ocr.service`: runs as the system user `meter-ocr` (sysusers.d), sandboxed (`ProtectSystem=strict`,
  `ProtectHome`, `PrivateTmp`).

## Installing

As with [ha-server-monitor-agent](https://github.com/ha-linux-monitoring/ha-server-monitor-agent), it ships through
a small custom pacman repository on the target host. Install and upgrade are then plain `pacman`.

```
./publish.sh <user>@<host>     # or: ./publish.sh local
```

This clones (or updates) this repository from GitHub into `~/meter-ocr-src`, builds the package with `makepkg`
and adds it to the local pacman repository `/srv/pacman-repo/friedpuppet/` (a second argument picks another
path; the repository is named after its last part). That directory must be writable by the user. Then, once,
add to `/etc/pacman.conf`:

```
[friedpuppet]
SigLevel = Optional TrustAll
Server = file:///srv/pacman-repo/friedpuppet
```

and install:

```
sudo pacman -Sy meter-ocr
```

The install scriptlet creates the user, then enables and starts the service. On first start the service pulls the
models it needs (`qwen3-vl:8b-instruct`, about 6 GB, and `glm-ocr`, about 3 GB) into Ollama; see
`journalctl -u meter-ocr`.

Open the port to the Home Assistant host only, for example:

```
sudo ufw allow from <HA address> to any port 8765 proto tcp
```

**Shipping an update**: bump `pkgver`/`pkgrel` in `PKGBUILD`, commit, push to `main`, run `./publish.sh` again, then
`sudo pacman -Syu`.

## Home Assistant side

Plain YAML, no custom integration:
- a `rest` sensor on `/status`;
- a `shell_command` that posts the queued photos with `curl -F` (`rest_command` can't send files);
- a `rest_command` that fetches `/batch/<id>`;
- a script that polls it and hands the results to the reading/voting script.

## License

MIT
