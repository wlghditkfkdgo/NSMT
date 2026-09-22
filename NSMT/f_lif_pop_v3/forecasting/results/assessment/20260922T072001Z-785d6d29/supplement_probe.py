from pathlib import Path
import sys,json,hashlib
root,out=map(Path,sys.argv[1:3])
p=Path(__file__).with_name('stage_probe.py')
exec(compile(p.read_text().split('cases=')[0],str(p),'exec'))
a,m,ds,dl,state=load('qk2-140541','seed7_flatten_spike_heterogeneous_sparse_qknorm_eta0.2_k3')
z=env['decompose'](m,dl,a)
r={'qk_trained_eta02_stages':z,'test_n_sequences':len(ds),'config':{k:getattr(a,k,None) for k in ['seed','data_seed','batch_size','n_train','n_val','n_test','epoch','max_eval_batches']},'split_sha256':{}}
for flag in ['train','val','test']:
 d=Dataset_Recall(a,flag);h=hashlib.sha256()
 for batch in DataLoader(d,batch_size=64):
  for x in batch:h.update(x.numpy().tobytes())
 r['split_sha256'][flag]=h.hexdigest()
out.write_text(json.dumps(r,indent=2));print(r)
