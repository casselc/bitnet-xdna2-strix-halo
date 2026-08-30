#!/usr/bin/env bash
# Deterministic pre-push scan for host-identifying or secret material in tracked
# text files. Not a DLP system -- a grep with a curated pattern list, meant to be
# run before every push to a PUBLIC repository.
#
# Exit 0 = clean, 1 = findings. Pass file paths to scan only those.
set -uo pipefail
cd "$(dirname "$0")/.."

HOST=$(hostname 2>/dev/null || echo __nohost__)
SHORTHOST=${HOST%%.*}
USERNAME=$(id -un 2>/dev/null || echo __nouser__)
HOME_PAT="/home/$USERNAME"

if [ $# -gt 0 ]; then FILES=("$@"); else mapfile -t FILES < <(git ls-files); fi

hits=0
report() { printf '  %-22s %s\n' "$1" "$2"; hits=$((hits+1)); }

for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    # skip binaries and the scanner's own pattern list
    file --mime-encoding "$f" 2>/dev/null | grep -q binary && continue
    case "$f" in tools/scan_artifacts.sh|tools/capture_environment.sh) continue;; esac

    while IFS= read -r m; do report "hostname" "$f:$m"; done < <(grep -nIF "$SHORTHOST" "$f" 2>/dev/null)
    while IFS= read -r m; do report "home-path" "$f:$m"; done < <(grep -nIF "$HOME_PAT" "$f" 2>/dev/null)
    # ZFS pool / dataset names and raw kernel cmdline
    while IFS= read -r m; do report "zfs/cmdline" "$f:$m"; done < <(grep -nIE 'root=ZFS=|rpool/|BOOT_IMAGE=' "$f" 2>/dev/null)
    # Routable-looking IPv4. Dotted version strings and decimal columns are the
    # common false positives, so octets are validated and version/firmware lines
    # skipped. Anything still reported needs a human to confirm.
    while IFS= read -r m; do report "ip-address?" "$f:$m"; done < <(python3 - "$f" <<'PYEOF'
import re, sys
pat = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
skip = re.compile(r"version|firmware|\bfw\b|release|revision", re.I)
for n, line in enumerate(open(sys.argv[1], errors="replace"), 1):
    if skip.search(line):
        continue
    for m in pat.finditer(line):
        o = [int(x) for x in m.groups()]
        if any(v > 255 for v in o):          # not an address at all
            continue
        if o[0] in (0, 127) or o[0] > 223:   # any/loopback/multicast/reserved
            continue
        print(f"{n}:{line.strip()[:120]}")
        break
PYEOF
)
    # credentials / tokens
    while IFS= read -r m; do report "possible-secret" "$f:$m"; done < <(
        grep -nIE '(ghp_|github_pat_|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY)' "$f" 2>/dev/null)
    while IFS= read -r m; do report "secret-assignment" "$f:$m"; done < <(
        grep -nIE '(API_?KEY|SECRET|PASSWORD|TOKEN)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{12,}' "$f" 2>/dev/null)
done

# large tracked files are usually accidental
while IFS= read -r f; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f")
    [ "$sz" -gt 5242880 ] && report "large-file" "$f ($((sz/1048576)) MiB)"
done < <(printf '%s\n' "${FILES[@]}")

if [ "$hits" -eq 0 ]; then echo "scan_artifacts: clean (${#FILES[@]} files)"; exit 0; fi
echo "scan_artifacts: $hits finding(s)"; exit 1
