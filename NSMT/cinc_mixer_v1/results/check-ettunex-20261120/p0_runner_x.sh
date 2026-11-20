#!/usr/bin/env bash
# p0_runner_x.sh (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning extension pre-registration): run_ett_spk.sh after the extension
# (PATCH 24 | 32, WD 0.005 | 0.02 | 0.05) against the runner of a base commit, without training. Same method as
# results/check-ettune-20261120/p0_runner.sh: a fake python records the argv of the training call; both runners use the real W.
#   A  default paths (no knob; 11 cases incl. population / SF-de / SEQ / MIXRESET): old argv == new argv
#   B  every value the base runner accepted (PATCH 4 16, EMB 32 128 256, MLPR, BLOCKS 2 3, WD 0 0.01 0.1, and all at their defaults):
#      old argv == new argv
#   C  each new value alone: new argv == default argv except the expected value tokens and the tag suffix
#   D  values outside the lists (PATCH 12 40, WD 0.03 0.050 0.2) and the new values on non-LIF conditions: exit 1, no training call
# usage (from cinc_mixer_v1/): bash results/check-ettunex-20261120/p0_runner_x.sh <base commit> > results/check-ettunex-20261120/p0_runner_x.txt
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
  grep -q "^W=\"$C\"$" "$f" || { echo "FAIL: could not fix W in $f"; exit 1; }
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
A=( "COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 -- ETTh2 96 42" "COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 -- ETTh2 720 7"
    "COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 -- ETTm2 96 2026" "COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 -- ETTm2 720 42"
    "COND=spk_ssa NALL=lif LR=0.0005 LAYERS=1 -- ETTh2 96 42" "COND=spk_ssa NALL=lif LR=0.0001 LAYERS=2 -- ETTm2 720 42"
    "COND=spkmx_pop_lif_muxlin_iand NSCOPE=mixer LR=0.0005 LAYERS=1 -- ETTh2 96 42"
    "COND=spkmx_pop_flif_muxlin_iand NSCOPE=mixer MIXPOOL=mean LR=0.0001 LAYERS=1 -- ETTm2 720 7"
    "COND=spkmx_pop_lif_muxlin_iand LR=0.0005 LAYERS=1 -- ETTh2 720 42" "COND=spk_ssa NALL=lif SEQ=336 LR=0.0005 LAYERS=1 -- ETTh2 96 42"
    "COND=spkmx_lif_muxlin_iand MIXRESET=zero LR=0.0005 LAYERS=1 -- ETTh2 96 42" )
i=0
for c in "${A[@]}"; do
  i=$((i+1)); argv "$T/old/run.sh" "$T/a_old$i" $c; argv "$T/new/run.sh" "$T/a_new$i" $c
  [ -s "$T/a_old$i" ] && cmp -s "$T/a_old$i" "$T/a_new$i"; ok $? "A$i default path, same argv ($(wc -l < "$T/a_new$i" 2>/dev/null) tokens): $c"
done
for ds in ETTh2 ETTm2; do
  d=0.5; [ $ds = ETTm2 ] && d=1
  for kv in "PATCH=8 EMB=64 MLPR=$d BLOCKS=1 WD=0.06" PATCH=4 PATCH=16 EMB=32 EMB=128 EMB=256 MLPR=0.5 MLPR=1 MLPR=2 MLPR=4 BLOCKS=2 BLOCKS=3 \
            WD=0 WD=0.01 WD=0.1 "PATCH=16 MLPR=2 WD=0.1" "PATCH=4 EMB=256 MLPR=4 BLOCKS=3 WD=0"; do
    argv "$T/old/run.sh" "$T/b_old" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 $kv -- $ds 720 7
    argv "$T/new/run.sh" "$T/b_new" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 $kv -- $ds 720 7
    [ -s "$T/b_old" ] && cmp -s "$T/b_old" "$T/b_new"; ok $? "B $ds $kv: same argv as the base runner"
  done
  argv "$T/old/run.sh" "$T/def_$ds" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 -- $ds 96 42
done
chk() {  # chk <ds> <env> <flag> <value> <tag suffix>
  local ds=$1 kv=$2 flag=$3 val=$4 suf=$5
  argv "$T/new/run.sh" "$T/c" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 $kv -- $ds 96 42
  python3 - "$T/def_$ds" "$T/c" "$flag" "$val" "$suf" <<'PYEOF'
import sys
a, b = open(sys.argv[1]).read().split('\n'), open(sys.argv[2]).read().split('\n')
flag, val, suf = sys.argv[3:6]
exp = list(a)
i = exp.index(flag); exp[i + 1] = val
t = exp.index('--tag'); exp[t + 1] = exp[t + 1] + suf
sys.exit(0 if exp == b else 1)
PYEOF
  ok $? "C $ds $kv -> $flag $val, tag +$suf, nothing else changed"
}
for ds in ETTh2 ETTm2; do
  chk $ds PATCH=24 --patch_size 24 _pt24; chk $ds PATCH=32 --patch_size 32 _pt32
  chk $ds WD=0.005 --weight_decay 0.005 _wd0.005; chk $ds WD=0.02 --weight_decay 0.02 _wd0.02; chk $ds WD=0.05 --weight_decay 0.05 _wd0.05
done
argv "$T/new/run.sh" "$T/c2" COND=spkmx_lif_muxlin_iand LR=0.0001 LAYERS=1 WD=0.02 MLPR=2 PATCH=32 -- ETTm2 96 42
grep -qx 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt32_mlp2_wd0.02' "$T/c2"; ok $? "C combined tag spkmx_lif_muxlin_iand_lr0.0001_L1_pt32_mlp2_wd0.02"
for c in "PATCH=12" "PATCH=40" "WD=0.03" "WD=0.050" "WD=0.2"; do
  argv "$T/new/run.sh" "$T/d" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 $c -- ETTh2 96 42
  [ "$(cat "$T/d.rc")" = 1 ] && [ ! -e "$T/d" ]; ok $? "D invalid $c: exit 1, no training call ($(head -1 "$T/d.msg"))"
done
for c in "COND=spk_ssa NALL=lif PATCH=24" "COND=spkmx_pop_lif_muxlin_iand NSCOPE=mixer WD=0.02" "COND=spkmx_flif_muxlin_iand PATCH=32"; do
  argv "$T/new/run.sh" "$T/d" $c LR=0.0005 LAYERS=1 -- ETTh2 96 42
  [ "$(cat "$T/d.rc")" = 1 ] && [ ! -e "$T/d" ]; ok $? "D new value on a non-LIF condition ($c): exit 1, no training call"
done
echo "$((n - fails))/$n PASS"
[ $fails = 0 ] && echo "P0 RUNNER X OK" || { echo "P0 RUNNER X FAILED"; exit 1; }
