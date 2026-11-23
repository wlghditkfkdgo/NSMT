#!/usr/bin/env bash
# p0_runner_sf.sh (2026-11-23, exp/f-lif-pop-v3; PROJECT_LOG SF-de patch-16 pre-registration): run_ett_spk.sh after allowing PATCH for
# COND=spk_ssa, against the runner of a base commit, without training (same method as check-ettunex-20261120/p0_runner_x.sh: a fake
# python records the argv of the training call; both runners use the real W).
#   A  paths that existed before (SF-de default 4 cells, SF-de SEQ 336, LIF default, the first-chain LIF tuned commands, population,
#      LIF PATCH 24): old argv == new argv
#   B  SF-de PATCH=16 (4 cells): new argv == the SF-de default argv except --patch_size 8 -> 16 and the tag ..._L<l>_nalif ->
#      ..._L<l>_pt16_nalif (both in the --tag argument and nowhere else); the base runner refused it (exit 1, no training call)
#   C  SF-de with any other knob (EMB 128, MLPR 2, BLOCKS 2, WD 0.1, PATCH 16 + WD 0.1) or an unlisted patch (12): exit 1, no training call
# usage (from cinc_mixer_v1/): bash results/check-sfpt16-20261123/p0_runner_sf.sh <base commit> > results/check-sfpt16-20261123/p0_runner_sf.txt
set -u
BASE=${1:?base commit}
C="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/old" "$T/new" "$T/py/bin" "$T/res" "$T/log"
git -C "$C" show "$BASE:NSMT/cinc_mixer_v1/scripts/run_ett_spk.sh" > "$T/old/run.sh"
cp "$C/scripts/run_ett_spk.sh" "$T/new/run.sh"
for f in "$T/old/run.sh" "$T/new/run.sh"; do
  [ "$(grep -c '^W=' "$f")" = 1 ] || { echo "FAIL: not exactly one W= line in $f"; exit 1; }
  sed -i "/^W=/c\\W=\"$C\"" "$f"
done
cat > "$T/py/bin/python" <<'PYEOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$ARGV_OUT"; exit 1
PYEOF
chmod +x "$T/py/bin/python"
fails=0; n=0
argv() {  # argv <runner> <outfile> <env...> -- <ds> <pl> <seed>
  local r=$1 o=$2; shift 2; local env=(); while [ "$1" != -- ]; do env+=("$1"); shift; done; shift
  rm -f "$o"; env "${env[@]}" ARGV_OUT="$o" PY="$T/py/bin/python" RES="$T/res" LOG="$T/log" SUITE=p0 bash "$r" "$@" 0 > "$o.msg" 2>&1; echo $? > "$o.rc"; }
ok() { n=$((n+1)); if [ "$1" = 0 ]; then echo "PASS $2"; else echo "FAIL $2"; fails=$((fails+1)); fi; }
A=( "COND=spk_ssa NALL=lif LR=0.0005 LAYERS=1 -- ETTh2 96 42" "COND=spk_ssa NALL=lif LR=0.0005 LAYERS=1 -- ETTh2 720 7"
    "COND=spk_ssa NALL=lif LR=0.0001 LAYERS=2 -- ETTm2 96 2026" "COND=spk_ssa NALL=lif LR=0.0001 LAYERS=2 -- ETTm2 720 42"
    "COND=spk_ssa NALL=lif SEQ=336 LR=0.0005 LAYERS=1 -- ETTh2 96 42"
    "COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 -- ETTh2 96 42"
    "COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 PATCH=16 WD=0.1 -- ETTh2 720 7"
    "COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 PATCH=16 MLPR=2 WD=0.1 -- ETTm2 96 2026"
    "COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 PATCH=24 EMB=256 BLOCKS=2 WD=0.02 -- ETTm2 720 42"
    "COND=spkmx_pop_lif_muxlin_iand NSCOPE=mixer LR=0.0005 LAYERS=1 -- ETTh2 96 42" )
i=0
for c in "${A[@]}"; do
  i=$((i+1)); argv "$T/old/run.sh" "$T/a_old$i" $c; argv "$T/new/run.sh" "$T/a_new$i" $c
  [ -s "$T/a_old$i" ] && cmp -s "$T/a_old$i" "$T/a_new$i"; ok $? "A$i existing path, same argv ($(wc -l < "$T/a_new$i" 2>/dev/null) tokens): $c"
done
for cell in "ETTh2 96 42 0.0005 1" "ETTh2 720 7 0.0005 1" "ETTm2 96 2026 0.0001 2" "ETTm2 720 42 0.0001 2"; do
  set -- $cell; ds=$1 pl=$2 s=$3 lr=$4 ly=$5
  argv "$T/new/run.sh" "$T/b_def" COND=spk_ssa NALL=lif LR=$lr LAYERS=$ly -- $ds $pl $s
  argv "$T/new/run.sh" "$T/b_new" COND=spk_ssa NALL=lif LR=$lr LAYERS=$ly PATCH=16 -- $ds $pl $s
  argv "$T/old/run.sh" "$T/b_old" COND=spk_ssa NALL=lif LR=$lr LAYERS=$ly PATCH=16 -- $ds $pl $s
  python3 - "$T/b_def" "$T/b_new" "spk_ssa_lr${lr}_L${ly}_nalif" "spk_ssa_lr${lr}_L${ly}_pt16_nalif" <<'PYEOF'
import sys
a, b = open(sys.argv[1]).read().split('\n'), open(sys.argv[2]).read().split('\n')
old_tag, new_tag = sys.argv[3], sys.argv[4]
ok = len(a) == len(b)
diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y] if ok else []
want = 0
for i, x, y in diff:
    if a[i - 1] == '--patch_size' and x == '8' and y == '16':
        want += 1
    elif a[i - 1] == '--tag' and x == old_tag and y == new_tag:
        want += 1
    else:
        ok = False
ok = ok and want == 2 and len(diff) == 2
print('ok' if ok else f'bad: {diff}')
sys.exit(0 if ok else 1)
PYEOF
  ok $? "B $ds H$pl seed $s: SF-de PATCH=16 argv = default argv except --patch_size 16 and the tag _pt16_nalif"
  [ "$(cat "$T/b_old.rc")" = 1 ] && [ ! -s "$T/b_old" ]; ok $? "B $ds H$pl: the base runner refused SF-de PATCH=16 (exit 1, no training call)"
done
for kv in EMB=128 MLPR=2 BLOCKS=2 WD=0.1 "PATCH=16 WD=0.1" PATCH=12; do
  argv "$T/new/run.sh" "$T/c" COND=spk_ssa NALL=lif LR=0.0005 LAYERS=1 $kv -- ETTh2 96 42
  [ "$(cat "$T/c.rc")" = 1 ] && [ ! -s "$T/c" ]; ok $? "C SF-de $kv: exit 1 before training ($(head -1 "$T/c.msg"))"
done
echo "$((n - fails))/$n PASS"
[ $fails = 0 ]
