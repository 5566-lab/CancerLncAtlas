from pathlib import Path
import socket,subprocess,json,sys
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');O=R/'kegg_cancer_redefinition_20261111_r1/fixed_y_remake_20261114_r1';P=O/'preview';P.mkdir(exist_ok=True)
sys.path.append(str(R/'python_deps'))
from PIL import Image,ImageDraw
pdf=next((O/'deliverables').glob('*.pdf'))
subprocess.run(['pdftoppm','-scale-to','1440','-png',str(pdf),str(P/'slide')],check=True)
files=sorted(P.glob('slide-*.png'));assert len(files)==17
sheet=Image.new('RGB',(1600,5*245),'#e6ebed');draw=ImageDraw.Draw(sheet)
for i,p in enumerate(files):
    im=Image.open(p).convert('RGB');im.thumbnail((390,220));x=(i%4)*400+5;y=(i//4)*245+20;sheet.paste(im,(x,y));draw.text((x,y-16),str(i+1),fill='black')
sheet.save(P/'CONTACT.png')
(O/'RENDER_RECEIPT.json').write_text(json.dumps(dict(target_host='149',pdf=str(pdf),rendered_pages=len(files),contact=str(P/'CONTACT.png')),indent=2)+'\n')
print('RENDERED_17_SLIDES',flush=True)
