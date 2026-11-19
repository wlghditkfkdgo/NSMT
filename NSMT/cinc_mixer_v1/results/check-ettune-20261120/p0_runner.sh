#!/usr/bin/env bash
# p0_runner.sh (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning pre-registration): run_ett_spk.sh with the tuning knobs
# (PATCH / EMB / MLPR / BLOCKS / WD) against the runner of a base commit, without training. A fake python records the argv of the
# training call (and exits 1, so the runner writes a SKIP line into a temporary RES); both runners are copied with W fixed to the real
# cinc_mixer_v1 folder, so paths in the argv are comparable.
#   A  default paths (no knob): old argv == new argv (11 cases, incl. population / SF-de / SEQ / MIXRESET)
#   B  every knob set to its default value: new argv == old default argv
#   C  each non-default value alone: new argv == default argv except the expected value tokens and the tag suffix
#   D  invalid values and knobs on non-LIF conditions: exit 1 before any training call
# usage (from cinc_mixer_v1/): bash results/check-ettune-20261120/p0_runner.sh <base commit> > results/check-ettune-20261120/p0_runner.txt
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
# A: default paths
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
# B: knobs at their default values
for ds in ETTh2 ETTm2; do
  m=0.5; [ $ds = ETTm2 ] && m=1
  argv "$T/old/run.sh" "$T/b_old_$ds" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 -- $ds 96 42
  argv "$T/new/run.sh" "$T/b_new_$ds" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 PATCH=8 EMB=64 MLPR=$m BLOCKS=1 WD=0.06 -- $ds 96 42
  cmp -s "$T/b_old_$ds" "$T/b_new_$ds"; ok $? "B $ds every knob at its default (PATCH=8 EMB=64 MLPR=$m BLOCKS=1 WD=0.06) = old default argv"
done
# C: one non-default value at a time; expected token changes
chk() {  # chk <ds> <env> <flag> <value> <tag suffix>
  local ds=$1 kv=$2 flag=$3 val=$4 suf=$5
  argv "$T/new/run.sh" "$T/c" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 $kv -- $ds 96 42
  python3 - "$T/b_old_$ds" "$T/c" "$flag" "$val" "$suf" <<'PYEOF'
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
  chk $ds PATCH=4 --patch_size 4 _pt4; chk $ds PATCH=16 --patch_size 16 _pt16
  chk $ds EMB=32 -emb 32 _e32; chk $ds EMB=128 -emb 128 _e128; chk $ds EMB=256 -emb 256 _e256
  for m in 0.5 1 2 4; do d=0.5; [ $ds = ETTm2 ] && d=1; [ $m = $d ] && continue; chk $ds MLPR=$m --mlp_ratios $m _mlp$m; done
  chk $ds BLOCKS=2 --layers 2 _blk2; chk $ds BLOCKS=3 --layers 3 _blk3
  chk $ds WD=0 --weight_decay 0 _wd0; chk $ds WD=0.01 --weight_decay 0.01 _wd0.01; chk $ds WD=0.1 --weight_decay 0.1 _wd0.1
done
# combined values: the tag order is fixed (_pt _e _mlp _blk _wd)
argv "$T/new/run.sh" "$T/c2" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 WD=0 BLOCKS=3 MLPR=4 EMB=256 PATCH=4 -- ETTh2 96 42
grep -qx 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt4_e256_mlp4_blk3_wd0' "$T/c2"; ok $? "C combined tag spkmx_lif_muxlin_iand_lr0.0005_L1_pt4_e256_mlp4_blk3_wd0"
# D: refusals before any training call
for c in "PATCH=5" "EMB=48" "MLPR=3" "MLPR=0.50" "BLOCKS=4" "WD=0.05" "WD=0.060"; do
  argv "$T/new/run.sh" "$T/d" COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 $c -- ETTh2 96 42
  [ "$(cat "$T/d.rc")" = 1 ] && [ ! -e "$T/d" ] && grep -q '^bad ' "$T/d.msg"; ok $? "D invalid $c: exit 1, no training call ($(cat "$T/d.msg"))"
done
for c in "COND=spk_ssa NALL=lif PATCH=4" "COND=spkmx_pop_lif_muxlin_iand NSCOPE=mixer EMB=128" "COND=spkmx_flif_muxlin_iand WD=0"; do
  argv "$T/new/run.sh" "$T/d" $c LR=0.0005 LAYERS=1 -- ETTh2 96 42
  [ "$(cat "$T/d.rc")" = 1 ] && [ ! -e "$T/d" ] && grep -q 'LIF SpkMixer conditions only' "$T/d.msg"; ok $? "D knob on a non-LIF condition ($c): exit 1, no training call"
done
echo "$((n-fails))/$n PASS"; [ $fails = 0 ] && echo "P0 RUNNER OK" || { echo "$fails FAILED"; exit 1; }
