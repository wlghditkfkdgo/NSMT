#!/usr/bin/env bash
# p0_mxett.sh <base commit> (2026-11-19 evening, exp/f-lif-pop-v3; mixer-only population ETT pre-registration): the existing paths are unchanged.
#   (1) make_queue_pn.sh: every earlier stage writes the same queue before (base commit) and after; mxett = the 48 ETT mixer-only lines of the
#       full 18:07 main stage with SUITE changed (scratch copies, results/ symlinked; the real queue files are not touched)
#   (2) summarize_pn.py: stages sf_ett, sf_mit, sf336, mxmit write the same summary before and after (--out to scratch)
# usage (from anywhere): bash p0_mxett.sh <base> > p0_mxett.txt
set -u
B=${1:?base commit}; W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; G="$(cd "$W/../.." && pwd)"
PY=/home/yschoi/.conda/envs/snn_recall/bin/python; export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
S=$(mktemp -d); mkdir -p $S/old/scripts/queues $S/new/scripts/queues $S/sum
(cd "$G" && git show $B:NSMT/cinc_mixer_v1/scripts/make_queue_pn.sh) > $S/old/scripts/make_queue_pn.sh; cp "$W/scripts/make_queue_pn.sh" $S/new/scripts/
ln -s "$W/results" $S/old/results; ln -s "$W/results" $S/new/results
echo "base $B, worktree HEAD $(cd "$G" && git rev-parse --short HEAD) + working tree"
for st in repro smoke sf main len336 sf336 mxmit; do
  o=$(bash $S/old/scripts/make_queue_pn.sh $st 2>&1 | tail -1); n=$(bash $S/new/scripts/make_queue_pn.sh $st 2>&1 | tail -1)
  cmp -s "${o%%:*}" "${n%%:*}" && r=same || r=DIFF; echo "make_queue_pn $st: old ${o##*:} lines, new ${n##*:} lines, $r"
done
n=$(bash $S/new/scripts/make_queue_pn.sh mxett | tail -1)
python3 - $S/new/scripts/queues/pn-main-20261118.txt "${n%%:*}" <<'PYEOF'
import sys
m = [l.strip().replace('SUITE=flif-v1-ett-spkmux-pn ', 'SUITE=flif-v1-ett-spkmux-mxpop ') for l in open(sys.argv[1]) if 'NSCOPE=mixer' in l and 'run_ett_spk' in l]
a = [l.strip() for l in open(sys.argv[2])]
print(f'make_queue_pn mxett: {len(a)} lines; the full 18:07 main stage has {len(m)} ETT mixer-only lines; same set with SUITE changed: {sorted(m) == sorted(a)}; unique: {len(set(a)) == len(a)}')
PYEOF
(cd "$G" && git show $B:NSMT/cinc_mixer_v1/summarize_pn.py) > $S/sum/summarize_pn.py
for st in sf_ett sf_mit sf336 mxmit; do
  (cd "$W" && PYTHONPATH="$W" $PY -B $S/sum/summarize_pn.py --stage $st --out $S/sum/$st.old.md > /dev/null 2>&1); a=$?
  (cd "$W" && $PY -B summarize_pn.py --stage $st --out $S/sum/$st.new.md > /dev/null 2>&1); b=$?
  cmp -s $S/sum/$st.old.md $S/sum/$st.new.md && r=same || r=DIFF; echo "summarize_pn --stage $st: rc old $a new $b, summary $r ($(wc -l < $S/sum/$st.old.md) lines)"
done
rm -rf $S
