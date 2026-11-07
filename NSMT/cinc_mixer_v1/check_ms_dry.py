"""Dry run of summarize_ms.py (2026-11-08, exp/f-lif-pop-v3): A rows are copies of the controls, so R = 0 and only the shortcut additions change the energy; then fail-closed cases. usage: python -B check_ms_dry.py <scratch dir>"""
import csv, os, re, shutil, subprocess, sys
C='/home/yschoi/CLS_spiking_transformer/wt-flif-pop-v3/NSMT/cinc_mixer_v1'
sys.path.insert(0,C)
R=C+'/results'
OUT=sys.argv[1]
PY=sys.executable
def rows(p):
    return [l.rstrip('\n') for l in open(p) if l.startswith('RESULT')]
def f(l,k):
    m=re.search(rf'\b{k}=(\S+)',l); return m[1] if m else None
LRM={'ETTh2':'0.0005','ETTm2':'0.0001'}; LRS={'ETTh2':'0.0005','ETTm2':'0.00025'}; LYS={'ETTh2':'1','ETTm2':'2'}
os.makedirs(OUT+'/ett',exist_ok=True); os.makedirs(OUT+'/mit',exist_ok=True)
raw=[]; en=[]
fields=['ds','pl','cond','tag','seed','mse_logged','total_fft_uJ','mse_recomputed','rate','binary','head_MAC','head_AC_corr','total_fft_uJ_corr','body_syn_uJ','body_syn_int_uJ','rule']
tune={ (e['tag'],e['ds'],e['pl'],e['seed']):e for e in csv.DictReader(open(R+'/flif-v1-ett-spkmux-tune/energy_recount.csv'))}
base={ (e['ds'],e['pl'],e['seed']):e for e in csv.DictReader(open(R+'/flif-v1-ett-spkmux/baseline_spk_ssa_energy.csv'))}
for l in rows(R+'/flif-v1-ett-spkmux-tune/raw.txt'):
    ds,tag=f(l,'ds'),f(l,'tag')
    if tag==f'spkmx_lif_muxlin_iand_lr{LRM[ds]}_L1':
        new=f'spkmx_lif_muxlin_ms_lr{LRM[ds]}_L1'
        raw.append(l.replace(f'tag={tag}',f'tag={new}').replace('cond=spkmx_lif_muxlin_iand','cond=spkmx_lif_muxlin_ms'))
        e=dict(tune[(tag,ds,f(l,'pl'),f(l,'seed'))]); e['tag']=new; e['cond']='spkmx_lif_muxlin_ms'; e['rule']='binary'; en.append(e)
    if ds=='ETTm2' and tag==f'spk_ssa_lr{LRS[ds]}_L{LYS[ds]}':
        new=tag+'_ms'; raw.append(l.replace(f'tag={tag}',f'tag={new}'))
        e=dict(tune[(tag,ds,f(l,'pl'),f(l,'seed'))]); e['tag']=new; e['rule']='binary'; en.append(e)
for l in rows(R+'/flif-v1-ett-spkmux/baseline_spk_ssa_raw.txt'):
    ds=f(l,'ds')
    if ds!='ETTh2': continue
    new=f'spk_ssa_lr{LRS[ds]}_L{LYS[ds]}_ms'
    raw.append(l+f' tag={new}')
    e=dict(base[(ds,f(l,'pl'),f(l,'seed'))]); e['tag']=new; e['rule']='binary'; en.append(e)
open(OUT+'/ett/raw.txt','w').write('\n'.join(raw)+'\n')
with open(OUT+'/ett/energy_recount.csv','w',newline='') as o:
    w=csv.DictWriter(o,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(en)
print('ett rows',len(raw),len(en))
# MIT-BIH
raw=[];en=[]
tm={ (e['tag'],e['seed'],e['val_fold']):e for e in csv.DictReader(open(R+'/flif-v1-mitbih-spkmux-tune/energy_recount.csv'))}
for l in rows(R+'/flif-v1-mitbih-spkmux-tune/raw.txt'):
    tag=f(l,'tag')
    if f(l,'val_fold')!='-1': continue
    if tag in ('spkmx_lif_muxlin_add_lr0.02_L2','spk_ssa_lr0.005_L2'):
        new={'spkmx_lif_muxlin_add_lr0.02_L2':'spkmx_lif_muxlin_ms_lr0.02_L2','spk_ssa_lr0.005_L2':'spk_ssa_lr0.005_L2_ms'}[tag]
        raw.append(l.replace(f'tag={tag}',f'tag={new}'))
        e=dict(tm[(tag,f(l,'seed'),'-1')]); e['tag']=new; e['rule']='binary'; en.append(e)
open(OUT+'/mit/raw.txt','w').write('\n'.join(raw)+'\n')
with open(OUT+'/mit/energy_recount.csv','w',newline='') as o:
    w=csv.DictWriter(o,fieldnames=list(en[0])); w.writeheader(); w.writerows(en)
print('mit rows',len(raw),len(en))
def run(task,suite,expect_rc):
    r=subprocess.run([PY,'-B','summarize_ms.py','--task',task,'--suite',suite,'--check_cells','x','--out',suite+'/summary.md'],cwd=C,capture_output=True,text=True)
    if r.returncode!=expect_rc:
        print('UNEXPECTED rc',r.returncode,'expected',expect_rc,task,suite); print(r.stdout[-800:]); print(r.stderr[-800:]); sys.exit(1)
    return r.stdout
o=run('ett',OUT+'/ett',0); print('\n'.join(x for x in o.split('\n') if x.startswith(('| M','| (ref'))))
o=run('mitbih',OUT+'/mit',0); print('\n'.join(x for x in o.split('\n') if x.startswith(('| M','| (ref'))))
# fail-closed cases
def case(name,task,src,edit):
    d=OUT+'/case'; shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d); edit(d); o=run(task,d,2); print('fail-closed ok:',name,'->',[x for x in o.split('\n') if x.startswith('STOP')][0][:90])
def drop(d):
    L=open(d+'/raw.txt').read().split('\n'); open(d+'/raw.txt','w').write('\n'.join(L[1:]))
def skip(d):
    open(d+'/raw.txt','a').write('SKIP(rc=1, no result) ds=ETTh2 pl=96 cond=spkmx_lif_muxlin_ms seed=42 out=x\n')
def nan(d):
    L=open(d+'/raw.txt').read().split('\n'); L[0]=re.sub(r' mse=\S+',' mse=nan',L[0]); open(d+'/raw.txt','w').write('\n'.join(L))
def badrec(d):
    t=list(csv.DictReader(open(d+'/energy_recount.csv'))); t[0]['mse_recomputed']='9.9'
    with open(d+'/energy_recount.csv','w',newline='') as o:
        w=csv.DictWriter(o,fieldnames=list(t[0])); w.writeheader(); w.writerows(t)
def nonbin(d):
    t=list(csv.DictReader(open(d+'/energy_recount.csv'))); t[0]['binary']='False'
    with open(d+'/energy_recount.csv','w',newline='') as o:
        w=csv.DictWriter(o,fieldnames=list(t[0])); w.writeheader(); w.writerows(t)
def dup(d):
    L=open(d+'/raw.txt').read().split('\n'); open(d+'/raw.txt','a').write(L[0]+'\n')
for n,fn in (('missing run',drop),('SKIP line',skip),('NaN metric',nan),('recount mismatch',badrec),('non-binary head input',nonbin),('duplicate',dup)):
    case(n,'ett',OUT+'/ett',fn)
def mdrop(d):
    L=open(d+'/raw.txt').read().split('\n'); open(d+'/raw.txt','w').write('\n'.join(L[1:]))
def mrule(d):
    t=list(csv.DictReader(open(d+'/energy_recount.csv'))); t[0]['rule']='integer'
    with open(d+'/energy_recount.csv','w',newline='') as o:
        w=csv.DictWriter(o,fieldnames=list(t[0])); w.writeheader(); w.writerows(t)
for n,fn in (('MIT-BIH missing run',mdrop),('MIT-BIH integer rule on a shortcut run',mrule)):
    case(n,'mitbih',OUT+'/mit',fn)
print('dry run ok')
