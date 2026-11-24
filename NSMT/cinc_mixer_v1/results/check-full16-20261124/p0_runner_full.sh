#!/usr/bin/env bash
# p0_runner_full.sh (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG full-ETT-table pre-registration): run_ett_spk.sh after allowing
# DS = ETTh1 | ETTm1 for the LIF SpkMixer, against the runner of a base commit, without training (same method as
# check-sfpt16-20261123/p0_runner_sf.sh: a fake python records the argv of the training call; both runners use the real W).
#   A  paths that existed before (SF-de default / PATCH 16 / SEQ 336, LIF default, both final LIF configurations, population):
#      old argv == new argv
#   B  ETTh1 / ETTm1 with both final configurations x H 96 / 192 / 336 / 720: new argv == the ETTh2 / ETTm2 argv of the same
#      command except the dataset name in --log_dir, --data and --data_path (3 tokens, nothing else); the base runner refused it
#   C  ETTh1 / ETTm1 with SF-de or a population condition, or an unknown dataset: exit 1, no training call
# usage (from cinc_mixer_v1/): bash results/check-full16-20261124/p0_runner_full.sh <base commit> > results/check-full16-20261124/p0_runner_full.txt
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
F1H="COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 PATCH=16 WD=0.1"
F1M="COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 PATCH=16 MLPR=2 WD=0.1"
FXH="COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 PATCH=24 WD=0.005"
FXM="COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 PATCH=24 EMB=256 BLOCKS=2 WD=0.02"
A=( "COND=spk_ssa NALL=lif LR=0.0005 LAYERS=1 -- ETTh2 96 42" "COND=spk_ssa NALL=lif LR=0.0001 LAYERS=2 -- ETTm2 720 42"
    "COND=spk_ssa NALL=lif LR=0.0005 LAYERS=1 PATCH=16 -- ETTh2 720 7" "COND=spk_ssa NALL=lif LR=0.0001 LAYERS=2 PATCH=16 -- ETTm2 96 2026"
    "COND=spk_ssa NALL=lif SEQ=336 LR=0.0005 LAYERS=1 -- ETTh2 96 42"
    "COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 -- ETTh2 96 42" "COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 -- ETTm2 720 7"
    "$F1H -- ETTh2 96 42" "$F1H -- ETTh2 720 7" "$F1M -- ETTm2 96 2026" "$F1M -- ETTm2 720 42"
    "$FXH -- ETTh2 96 42" "$FXH -- ETTh2 192 7" "$FXM -- ETTm2 336 2026" "$FXM -- ETTm2 720 42"
    "COND=spkmx_pop_lif_muxlin_iand NSCOPE=mixer LR=0.0005 LAYERS=1 -- ETTh2 96 42" )
i=0
for c in "${A[@]}"; do
  i=$((i+1)); argv "$T/old/run.sh" "$T/a_old$i" $c; argv "$T/new/run.sh" "$T/a_new$i" $c
  [ -s "$T/a_old$i" ] && cmp -s "$T/a_old$i" "$T/a_new$i"; ok $? "A$i existing path, same argv ($(wc -l < "$T/a_new$i" 2>/dev/null) tokens): $c"
done
for pair in "ETTh1 ETTh2" "ETTm1 ETTm2"; do
  set -- $pair; nd=$1; od=$2
  for cfg in 1 X; do
    case $od$cfg in ETTh21) e=$F1H ;; ETTm21) e=$F1M ;; ETTh2X) e=$FXH ;; ETTm2X) e=$FXM ;; esac
    for pl in 96 192 336 720; do
      argv "$T/new/run.sh" "$T/b_ref" $e -- $od $pl 42
      argv "$T/new/run.sh" "$T/b_new" $e -- $nd $pl 42
      argv "$T/old/run.sh" "$T/b_old" $e -- $nd $pl 42
      python3 - "$T/b_ref" "$T/b_new" "$od" "$nd" <<'PYEOF'
import sys
a, b = open(sys.argv[1]).read().split('\n'), open(sys.argv[2]).read().split('\n')
od, nd = sys.argv[3], sys.argv[4]
ok = len(a) == len(b) and len(a) > 20
diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y] if ok else []
want = set()
for i, x, y in diff:
    if a[i - 1] in ('--log_dir', '--data', '--data_path') and x.replace(od, nd) == y and x.count(od) == 1:
        want.add(a[i - 1])
    else:
        ok = False
ok = ok and want == {'--log_dir', '--data', '--data_path'} and len(diff) == 3
print('ok' if ok else f'bad: {diff}')
sys.exit(0 if ok else 1)
PYEOF
      ok $? "B $nd H$pl config $cfg: argv = the $od argv except the dataset name in --log_dir / --data / --data_path"
      [ "$(cat "$T/b_old.rc")" = 1 ] && [ ! -s "$T/b_old" ]; ok $? "B $nd H$pl config $cfg: the base runner refused it (exit 1, no training call)"
    done
  done
done
for c in "COND=spk_ssa NALL=lif LR=0.0005 LAYERS=1 -- ETTh1 96 42" "COND=spk_ssa NALL=lif LR=0.0001 LAYERS=2 PATCH=16 -- ETTm1 96 42"
do argv "$T/new/run.sh" "$T/c" $c; [ "$(cat "$T/c.rc")" = 1 ] && [ ! -s "$T/c" ]; ok $? "C refused before training ($(head -1 "$T/c.msg")): $c"; done
for c in "COND=spkmx_pop_lif_muxlin_iand NSCOPE=mixer LR=0.0005 LAYERS=1 -- ETTh1 96 42" "$F1H -- ETTh3 96 42" "$F1H -- electricity 96 42"
do argv "$T/new/run.sh" "$T/c" $c; [ "$(cat "$T/c.rc")" = 1 ] && [ ! -s "$T/c" ]; ok $? "C refused before training ($(head -1 "$T/c.msg")): $c"; done
echo "$((n - fails))/$n PASS"
[ $fails = 0 ]
