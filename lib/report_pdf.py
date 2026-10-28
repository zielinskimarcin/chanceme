"""Render the current verified comparison as a downloadable, paginated PDF."""
import io,json,math,sys
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,Color,white
from reportlab.lib.pagesizes import landscape,A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

import reportlab
FONT_ROOT=Path(reportlab.__file__).parent/'fonts'
REG=str(FONT_ROOT/'Vera.ttf')
BOLD=str(FONT_ROOT/'VeraBd.ttf')
for name,path in [('UI',REG),('UI-Bold',BOLD)]:
    pdfmetrics.registerFont(TTFont(name,path))
INK=HexColor('#17324d');MUTED=HexColor('#58718c');GREEN=HexColor('#72b6a5');RED=HexColor('#ce7489');LINE=HexColor('#e3ecf3')

def render(payload):
    profile=payload['profile'];estimate=payload['estimate'];comparison=payload['comparison'];records=comparison['records'];counts=comparison['nearby'];filters=comparison['filters']
    p=estimate['probability'];ring=HexColor('#cc8392' if p<.35 else '#dbb66a' if p<.65 else '#8fc7b4')
    records=sorted(records,key=lambda r:abs(r['score']-profile['score'])/counts['test_window']+abs(r['gpa']-profile['gpa'])/.5)
    out=io.BytesIO();c=canvas.Canvas(out,pagesize=landscape(A4));W,H=landscape(A4)
    c.setTitle('Chance Me - Bocconi admission report');c.setAuthor('Chance Me')
    def text(x,y,t,size=9,bold=False,color=INK):
        c.setFillColor(color);c.setFont('UI-Bold' if bold else 'UI',size);c.drawString(x,y,str(t))
    def wrap(t,x,y,width,size=9,bold=False,leading=None):
        words=str(t).split();line='';leading=leading or size*1.35
        for word in words:
            trial=(line+' '+word).strip()
            if pdfmetrics.stringWidth(trial,'UI-Bold' if bold else 'UI',size)>width and line:
                text(x,y,line,size,bold);y-=leading;line=word
            else:line=trial
        if line:text(x,y,line,size,bold)
        return y-leading
    def box(x,y,w,h,color,border=LINE,r=12):
        c.setFillColor(color);c.setStrokeColor(border);c.setLineWidth(.6);c.roundRect(x,y,w,h,r,fill=1,stroke=1)
    def year(v):return '20'+v[:2]+'/'+v[-2:]
    def number(v):return f'{v:g}'
    page_count=1+math.ceil(max(0,len(records)-4)/22)
    def footer(page):
        text(32,20,'Self-reported outcomes. Independent tool; not affiliated with Bocconi.',7,color=MUTED)
        text(W-98,20,f'{page} / {page_count}',7,color=MUTED)
    def table(rows,top):
        xs=[46,126,196,280,385,575]
        box(32,top-20,W-64,20,HexColor('#edf5fb'),r=5)
        for x,h in zip(xs,['SAT' if profile['test_type']=='SAT' else 'Bocconi Test','GPA','Cycle','Round','Reported outcome','Match']):text(x,top-13,h,7.5,True,color=MUTED)
        y=top-20
        for i,r in enumerate(rows):
            if i%2==1:c.setFillColor(HexColor('#f7fafc'));c.rect(32,y-21,W-64,21,fill=1,stroke=0)
            text(xs[0],y-14,number(r['score']),9,True);text(xs[1],y-14,number(r['gpa']),9)
            text(xs[2],y-14,year(r['cycle']),8.5);text(xs[3],y-14,r['session'],8.5)
            text(xs[4],y-14,'Admitted' if r['outcome']=='admitted' else 'Not admitted',8.5,True,color=HexColor('#326a57') if r['outcome']=='admitted' else HexColor('#a24d63'))
            text(xs[5],y-14,('Similar' if r['nearby'] else 'Outside range')+(' / Comparison only' if r.get('used_for_bt_fit') is False else ''),8,color=MUTED);y-=21
        return y
    text(32,H-43,'Your result, in context',23,True);text(W-120,H-39,'chance me',14,True)
    y=H-145;box(32,y,W-64,78,HexColor('#f1f7fc'))
    for x,label,value,width in [(47,'SAT score' if profile['test_type']=='SAT' else 'Bocconi Test',number(profile['score']),65),(120,'Bocconi GPA',(number(profile['gpa']) if '.' in number(profile['gpa']) else number(profile['gpa'])+'.0'),60),(194,'Programme',profile['course'],184),(392,'Round',profile['session'],63)]:
        text(x,y+57,label,7,color=MUTED);wrap(value,x,y+36,width,10 if label!='Programme' or len(value)<35 else 8.5,True)
    cx=510;cy=y+40;radius=27
    c.setLineWidth(6);c.setStrokeColor(HexColor('#deecee'));c.circle(cx,cy,radius,fill=0,stroke=1)
    c.setStrokeColor(ring);c.setLineCap(1)
    c.arc(cx-radius,cy-radius,cx+radius,cy+radius,startAng=90,extent=-p*360)
    c.setFillColor(INK);c.setFont('UI-Bold',17);c.drawCentredString(cx,cy-6,estimate['display_percent'])
    text(553,y+49,'GPA simulation' if profile.get('gpa_estimated') else 'Estimated chance',10,True);wrap('A model estimate, not an admission decision.',553,y+31,205,8)
    chosen=filters['course']+' / '+('Both rounds' if filters['session']=='All' else filters['session'])+' / '+(year(filters['cycle']) if filters.get('cycle') else 'All cycles')
    text(32,y-22,chosen,8.5,True)
    text(32,y-35,comparison.get('proxy_notice',''),6.5,color=MUTED)
    chart_y=H-399;chart_h=210;chart_w=510
    box(32,chart_y,chart_w,chart_h,white);text(46,chart_y+chart_h-20,'Where your profile sits',10,True)
    left=76;right=522;bottom=chart_y+36;top=chart_y+chart_h-39
    scores=[r['score'] for r in records]+[profile['score']];gpas=[r['gpa'] for r in records]+[profile['gpa']]
    step=100 if profile['test_type']=='SAT' else 5
    xmin=max(400 if profile['test_type']=='SAT' else -11.17,math.floor((min(scores)-step*.4)/step)*step)
    xmax=min(1600 if profile['test_type']=='SAT' else 50,math.ceil((max(scores)+step*.4)/step)*step)
    ymin=max(6,math.floor((min(gpas)-.2)*2)/2);ymax=min(10,math.ceil((max(gpas)+.2)*2)/2)
    X=lambda n:left+(n-xmin)/(xmax-xmin)*(right-left)
    Y=lambda n:bottom+(n-ymin)/(ymax-ymin)*(top-bottom)
    c.setLineWidth(.5)
    for i in range(5):
        xx=left+(right-left)*i/4;v=xmin+(xmax-xmin)*i/4;c.setStrokeColor(LINE);c.line(xx,bottom,xx,top);text(xx-10,bottom-13,f'{v:.0f}',7,color=MUTED)
        yy=bottom+(top-bottom)*i/4;v=ymin+(ymax-ymin)*i/4;c.line(left,yy,right,yy);text(left-27,yy-2,f'{v:.1f}',7,color=MUTED)
    for r in sorted(records,key=lambda r:r['outcome']=='rejected'):
        xx=X(r['score']);yy=Y(r['gpa'])
        if r['outcome']=='admitted':c.setFillColor(GREEN);c.circle(xx,yy,2.5,fill=1,stroke=0)
        else:c.setStrokeColor(RED);c.setLineWidth(1.2);c.line(xx-2,yy-2,xx+2,yy+2);c.line(xx-2,yy+2,xx+2,yy-2)
    xx=X(profile['score']);yy=Y(profile['gpa']);c.setFillColor(white);c.setStrokeColor(INK);c.setLineWidth(1.5)
    path=c.beginPath();path.moveTo(xx,yy+5);path.lineTo(xx+5,yy);path.lineTo(xx,yy-5);path.lineTo(xx-5,yy);path.close();c.drawPath(path,fill=1,stroke=1);text(xx-6,yy+10,'You',7,True)
    c.saveState();c.translate(46,(top+bottom)/2);c.rotate(90);text(-24,0,'Bocconi GPA',7,color=MUTED);c.restoreState()
    text(226,chart_y+8,'SAT score' if profile['test_type']=='SAT' else 'Bocconi Test score',7,color=MUTED)
    box(558,chart_y,W-590,chart_h,HexColor('#edf7f2'))
    text(575,chart_y+chart_h-24,'Similar profiles',10,True)
    text(575,chart_y+chart_h-65,(str(counts['admitted'])+' of '+str(counts['total'])) if counts['total'] else '0',26,True)
    text(575,chart_y+chart_h-84,'reported admission' if counts['total'] else 'similar reports',9,True)
    text(575,chart_y+72,f"{counts['admitted']}  Admitted",10,True,color=HexColor('#326a57'));text(575,chart_y+48,f"{counts['rejected']}  Not admitted",10,True,color=HexColor('#a24d63'))
    wrap('This share is not your admission probability.' if counts['total'] else 'Reports outside this range are listed below.',575,chart_y+25,W-625,7)
    text(32,chart_y-21,'Individual reports',10,True)
    table(records[:4],chart_y-33);footer(1);c.showPage()
    for page,start in enumerate(range(4,len(records),22),2):
        text(32,H-42,'Individual reports',20,True);text(32,H-61,chosen,8.5,color=MUTED)
        table(records[start:start+22],H-80);footer(page);c.showPage()
    c.save();return out.getvalue()

if __name__=='__main__':
    sys.stdout.buffer.write(render(json.load(sys.stdin)))
