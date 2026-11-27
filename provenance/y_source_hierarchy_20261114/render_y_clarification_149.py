from pathlib import Path
import socket,subprocess,json,sys,math
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');O=R/'kegg_cancer_redefinition_20261111_r1/ppt_y_clarification_20261114_r2';D=O/'deliverables';P=O/'preview';P.mkdir(exist_ok=True)
sys.path.append(str(R/'python_deps'))
from PIL import Image,ImageDraw
manifest=json.loads((O/'PPT_MANIFEST.json').read_text());ppt=Path(manifest['pptx']);profile=O/'libreoffice_profile'
(O/'RENDER_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'Render and inspect extended presentation'},indent=2))
subprocess.run(['libreoffice','-env:UserInstallation='+profile.as_uri(),'--headless','--convert-to','pdf','--outdir',str(D),str(ppt)],check=True,timeout=120)
pdf=ppt.with_suffix('.pdf');assert pdf.exists()
subprocess.run(['pdftoppm','-scale-to','1440','-png',str(pdf),str(P/'slide')],check=True,timeout=120)
files=sorted(P.glob('slide-*.png'));assert len(files)==manifest['slides']
sheet=Image.new('RGB',(1600,math.ceil(len(files)/4)*245),'#e6ebed');draw=ImageDraw.Draw(sheet)
for i,p in enumerate(files):
    im=Image.open(p).convert('RGB');im.thumbnail((390,220));x=(i%4)*400+5;y=(i//4)*245+20;sheet.paste(im,(x,y));draw.text((x,y-16),str(i+1),fill='black')
sheet.save(P/'CONTACT.png')
(O/'RENDER_RECEIPT.json').write_text(json.dumps({'target_host':'149','pdf':str(pdf),'rendered_pages':len(files),'contact':str(P/'CONTACT.png')},indent=2)+'\n')
print('RENDERED',len(files),'SLIDES',flush=True)
