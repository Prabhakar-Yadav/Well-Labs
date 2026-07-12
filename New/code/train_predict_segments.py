# Train a model on the hand-labelled TLBC reaches (clear/partial/blocked) + ortho,
# then predict the aligned segments and map to 4 classes:
#   clear->clear, blocked->blocked, model-partial split by % -> partial(<=30)/moderate(>30).
# Corridor = 0.5 + width/2.  python train_predict_segments.py
import glob, numpy as np, rasterio, cv2, shapefile, collections
from rasterio.windows import Window
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
ORTHOS=['_keep_locally/TLBC_D95_ATTANUR_2_ortho.tif','_keep_locally/TLBC_D95_ATTANUR_3_ortho.tif',
        '_keep_locally/TLBC_D95_ATTANUR_1_ortho.tif','_keep_locally/TLBC_D95_ATTANUR_4_ortho.tif',
        '_keep_locally/TLBC_D95_SHAKAPUR_ortho.tif']
WIDTH={'main':14.0,'distributary':5.0,'lateral':3.3}; RANK={'clear':0,'partial':1,'blocked':2}
def norm(s):
    s=str(s).strip().lower(); return 'blocked' if 'block' in s else 'partial' if 'part' in s else 'clear' if 'clear' in s else None
def feats(rgb,mk,mc):
    cp=mk>0
    if cp.sum()<15: return None
    r,g,b=[rgb[:,:,i].astype(np.float32) for i in range(3)]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); h,s,v=[hsv[:,:,i].astype(np.float32) for i in range(3)]
    exg=2*g-r-b; vari=(g-r)/(g+r-b+1e-6); green=(h>=35)&(h<=85)&(s>40)&(v>50)
    veg=((exg>25)|((vari>0.1)&green)); water=(v<120)&(b>r)&(s>15); bright=(v>180)
    gray=0.299*r+0.587*g+0.114*b; cp2=cp
    fr=lambda c,m:float((c&(m>0)).sum())/max(1,int((m>0).sum()))
    return [fr(veg,mk),fr(water,mk),fr(bright,mk),float(exg[cp2].mean()),float(s[cp2].mean()),
            float(v[cp2].mean()),float(gray[cp2].std()),float(vari[cp2].mean()),fr(veg,mc),fr(~veg,mc)]
def load(path,wfn):
    r=shapefile.Reader(path); f=[x[0] for x in r.fields[1:]]; fld=list(r.fields[1:])
    it=[dict(rec=dict(zip(f,sr.record)),pts=np.array(sr.shape.points,float),width=wfn(dict(zip(f,sr.record))),feat=None,veg=None) for sr in r.shapeRecords()]
    r.close(); return it,fld
train,_=load('outputs/07_blockage_finetuned/tlbc_blockage_reaches.shp',lambda d:WIDTH.get(str(d['ctype']).strip(),5.0))
segs,segfld=load('gis/TLBC_alignment/tlbc_segments.shp',lambda d:float(d['width']))
ALL=train+segs; print(f'train {len(train)} + segments {len(segs)}',flush=True)
for op in ORTHOS:
    if not glob.glob(op): continue
    if not [it for it in ALL if it['feat'] is None]: break
    with rasterio.open(op) as src:
        H,W=src.height,src.width; tr=src.transform; px_m=abs(tr.a); b=src.bounds
        for it in [x for x in ALL if x['feat'] is None]:
            pts=it['pts']; cx,cy=pts[:,0].mean(),pts[:,1].mean()
            if not (b.left<=cx<=b.right and b.bottom<=cy<=b.top): continue
            hw=max(2,int((0.5+it['width']/2)/px_m)); hc=max(1,int(1.0/px_m))
            rr,cc=rasterio.transform.rowcol(tr,pts[:,0].tolist(),pts[:,1].tolist()); rr,cc=np.array(rr),np.array(cc)
            ins=(rr>=0)&(rr<H)&(cc>=0)&(cc<W); rr,cc=rr[ins],cc[ins]
            if len(rr)<2: continue
            r0=max(0,rr.min()-hw); r1=min(H,rr.max()+hw); a0=max(0,cc.min()-hw); a1=min(W,cc.max()+hw)
            if r1-r0<4 or a1-a0<4: continue
            d=src.read([1,2,3,4],window=Window(a0,r0,a1-a0,r1-r0)); valid=d[3]==255
            if valid.mean()<0.3: continue
            rgb=np.transpose(d[:3],(1,2,0)); loc=np.array([(cc[i]-a0,rr[i]-r0) for i in range(len(rr))],np.int32)
            mk=np.zeros(rgb.shape[:2],np.uint8); mc=np.zeros_like(mk)
            cv2.polylines(mk,[loc],False,255,2*hw); mk=((mk>0)&valid).astype(np.uint8); cv2.polylines(mc,[loc],False,255,2*hc)
            fv=feats(rgb,mk,mc)
            if fv is not None: it['feat']=fv; it['veg']=fv[0]
        print(f'{op.split("/")[-1]}: train {sum(1 for t in train if t["feat"])}/{len(train)}, seg {sum(1 for s in segs if s["feat"])}/{len(segs)}',flush=True)
Xtr=np.array([t['feat'] for t in train if t['feat'] and norm(t['rec']['status'])])
ytr=np.array([norm(t['rec']['status']) for t in train if t['feat'] and norm(t['rec']['status'])])
print('training samples:',len(Xtr),dict(collections.Counter(ytr.tolist())),flush=True)
skf=StratifiedKFold(5,shuffle=True,random_state=0); best=None
for nm,mdl in [('HistGB',HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=0)),
               ('RF',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=0))]:
    cvp=cross_val_predict(mdl,Xtr,ytr,cv=skf); ex=100*(cvp==ytr).mean()
    wi=100*np.mean([abs(RANK[p]-RANK[t])<=1 for p,t in zip(cvp,ytr)]); bina=100*np.mean([(p=='clear')==(t=='clear') for p,t in zip(cvp,ytr)])
    print(f'  {nm}: exact {ex:.1f}%  within-1 {wi:.1f}%  clear-vs-blocked {bina:.1f}%',flush=True)
    if best is None or ex>best[1]: best=(nm,ex,mdl)
nm,ex,mdl=best; mdl.fit(Xtr,ytr); print('BEST:',nm,f'{ex:.1f}% exact (3-class CV)',flush=True)
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
print('4-class predicted:',dict(dist),'| not-sampled:',nulls); print('done -> gis/TLBC_alignment/tlbc_segments.shp')
