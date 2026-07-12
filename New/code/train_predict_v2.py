# Stronger model: 18 features (veg/water/soil/bright + texture + edges + multi-scale),
# compares RF / ExtraTrees / HistGB / MLP(neural net) / Voting ensemble by 5-fold CV,
# picks the best, predicts TLBC segments -> 4 class. Caches features to npz.
import glob, numpy as np, rasterio, cv2, shapefile, collections
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
ORTHOS=['_keep_locally/TLBC_D95_ATTANUR_2_ortho.tif','_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif',
        '_keep_locally/TLBC_D95_ATTANUR_1_ortho.tif','_keep_locally/TLBC_D95_ATTANUR_4_ortho.tif',
        '_keep_locally/TLBC_D95_SHAKAPUR_ortho.tif']
WIDTH={'main':14.0,'distributary':5.0,'lateral':3.3}; RANK={'clear':0,'partial':1,'blocked':2}
def norm(s):
    s=str(s).strip().lower(); return 'blocked' if 'block' in s else 'partial' if 'part' in s else 'clear' if 'clear' in s else None
def feats(rgb,mk,mc1,mc2):
    cp=mk>0
    if cp.sum()<15: return None
    r,g,b=[rgb[:,:,i].astype(np.float32) for i in range(3)]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); h,s,v=[hsv[:,:,i].astype(np.float32) for i in range(3)]
    exg=2*g-r-b; vari=(g-r)/(g+r-b+1e-6); green=(h>=35)&(h<=85)&(s>40)&(v>50)
    veg=((exg>25)|((vari>0.1)&green)); water=(v<120)&(b>r)&(s>15); bright=(v>180); soil=(s<60)&(v>=120)&(v<=200)&(r>g)
    gray=(0.299*r+0.587*g+0.114*b).astype(np.uint8)
    edges=cv2.Canny(gray,50,150); sob=np.hypot(cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,1,0,ksize=3),cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,0,1,ksize=3))
    fr=lambda c,m:float((c&(m>0)).sum())/max(1,int((m>0).sum()))
    return [fr(veg,mk),fr(water,mk),fr(bright,mk),fr(soil,mk),
            float(exg[cp].mean()),float(exg[cp].std()),float(s[cp].mean()),float(v[cp].mean()),
            float(gray[cp].std()),float(vari[cp].mean()),
            fr(veg,mc1),fr(~veg,mc1),fr(water,mc1),fr(veg,mc2),fr(veg,mk)-fr(veg,mc1),
            float((edges[cp]>0).mean()),float(sob[cp].mean()),float(h[cp].std())]
def load(path,wfn):
    r=shapefile.Reader(path); f=[x[0] for x in r.fields[1:]]; fld=list(r.fields[1:])
    it=[dict(rec=dict(zip(f,sr.record)),pts=np.array(sr.shape.points,float),width=wfn(dict(zip(f,sr.record))),feat=None,veg=None) for sr in r.shapeRecords()]
    r.close(); return it,fld
train,_=load('outputs/07_blockage_finetuned/tlbc_blockage_reaches.shp',lambda d:WIDTH.get(str(d['ctype']).strip(),5.0))
segs,segfld=load('gis/TLBC_alignment/tlbc_segments.shp',lambda d:float(d['width'])); ALL=train+segs
print(f'train {len(train)} + segments {len(segs)}',flush=True)
for op in ORTHOS:
    if not glob.glob(op) or not [x for x in ALL if x['feat'] is None]: continue
    with rasterio.open(op) as src:
        H,W=src.height,src.width; tr=src.transform; px_m=abs(tr.a); b=src.bounds
        for it in [x for x in ALL if x['feat'] is None]:
            pts=it['pts']; cx,cy=pts[:,0].mean(),pts[:,1].mean()
            if not (b.left<=cx<=b.right and b.bottom<=cy<=b.top): continue
            hw=max(2,int((0.5+it['width']/2)/px_m)); h1=max(1,int(1.0/px_m)); h2=max(1,int(2.0/px_m))
            rr,cc=rasterio.transform.rowcol(tr,pts[:,0].tolist(),pts[:,1].tolist()); rr,cc=np.array(rr),np.array(cc)
            ins=(rr>=0)&(rr<H)&(cc>=0)&(cc<W); rr,cc=rr[ins],cc[ins]
            if len(rr)<2: continue
            r0=max(0,rr.min()-hw); r1=min(H,rr.max()+hw); a0=max(0,cc.min()-hw); a1=min(W,cc.max()+hw)
            if r1-r0<4 or a1-a0<4: continue
            d=src.read([1,2,3,4],window=Window(a0,r0,a1-a0,r1-r0)); valid=d[3]==255
            if valid.mean()<0.3: continue
            rgb=np.transpose(d[:3],(1,2,0)); loc=np.array([(cc[i]-a0,rr[i]-r0) for i in range(len(rr))],np.int32)
            mk=np.zeros(rgb.shape[:2],np.uint8); mc1=np.zeros_like(mk); mc2=np.zeros_like(mk)
            cv2.polylines(mk,[loc],False,255,2*hw); mk=((mk>0)&valid).astype(np.uint8)
            cv2.polylines(mc1,[loc],False,255,2*h1); cv2.polylines(mc2,[loc],False,255,2*h2)
            fv=feats(rgb,mk,mc1,mc2)
            if fv is not None: it['feat']=fv; it['veg']=fv[0]
        print(f'{op.split("/")[-1]}: train {sum(1 for t in train if t["feat"])}/{len(train)}, seg {sum(1 for s in segs if s["feat"])}/{len(segs)}',flush=True)
Xtr=np.array([t['feat'] for t in train if t['feat'] and norm(t['rec']['status'])]); ytr=np.array([norm(t['rec']['status']) for t in train if t['feat'] and norm(t['rec']['status'])])
np.savez('New/results/_tlbc_feats.npz',Xtr=Xtr,ytr=ytr,Xseg=np.array([s['feat'] if s['feat'] else [np.nan]*18 for s in segs]))
print('training samples:',len(Xtr),dict(collections.Counter(ytr.tolist())),flush=True)
skf=StratifiedKFold(5,shuffle=True,random_state=0)
rf=RandomForestClassifier(400,min_samples_leaf=2,class_weight='balanced',random_state=0)
et=ExtraTreesClassifier(400,min_samples_leaf=2,class_weight='balanced',random_state=0)
hg=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=0)
mlp=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-2,max_iter=2000,random_state=0))
vote=VotingClassifier([('rf',rf),('hg',hg),('mlp',mlp)],voting='soft')
models={'RF':rf,'ExtraTrees':et,'HistGB':hg,'MLP(neural net)':mlp,'Voting ensemble':vote}
best=None
for nm,mdl in models.items():
    cvp=cross_val_predict(mdl,Xtr,ytr,cv=skf); ex=100*(cvp==ytr).mean()
    wi=100*np.mean([abs(RANK[p]-RANK[t])<=1 for p,t in zip(cvp,ytr)]); bina=100*np.mean([(p=='clear')==(t=='clear') for p,t in zip(cvp,ytr)])
    print(f'  {nm:16} exact {ex:.1f}%  within-1 {wi:.1f}%  clear-vs-blocked {bina:.1f}%',flush=True)
    if best is None or ex>best[1]: best=(nm,ex,mdl)
nm,ex,mdl=best; mdl.fit(Xtr,ytr); print('BEST:',nm,f'{ex:.1f}% exact',flush=True)
def to4(c3,pct): return 'clear' if c3=='clear' else 'blocked' if c3=='blocked' else ('partial' if pct<=30 else 'moderate')
w=shapefile.Writer('gis/TLBC_alignment/tlbc_segments',shapeType=shapefile.POLYLINE)
for fd in segfld: w.field(*fd)
dist=collections.Counter(); nulls=0
for sg in segs:
    d=sg['rec']
    if sg['feat'] is None: pred=''; pct=0; nulls+=1
    else: c3=str(mdl.predict([sg['feat']])[0]); pct=int(round(sg['veg']*100)); pred=to4(c3,pct); dist[pred]+=1
    w.line([[[float(x),float(y)] for x,y in sg['pts']]])
    w.record(d['id'],d['canal'],d['ctype'],d['width'],d['chainage'],pred,str(d.get('actual','')).strip(),pct)
w.close()
print('4-class predicted:',dict(dist),'| not-sampled:',nulls); print('done')
